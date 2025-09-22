from __future__ import annotations

import copy

import inspect

import itertools
import logging
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch import nn
from tqdm import tqdm

from . import positional_encodings, priors, utils
from .bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_limits,
    get_custom_bar_dist,
)
from .priors import prior
from .transformer import TransformerModel
from .utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    get_uniform_single_eval_pos_sampler,
    get_weighted_single_eval_pos_sampler,
    init_dist,
    StoreDictKeyPair,
)
import wandb
import os
import json

class ProbabilisticRankingLoss(nn.Module):
    
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, output, y_true, task_ids):
        mu = output[..., 0]
        sigma = output[..., 1].exp().sqrt()
        
        assert torch.all(task_ids[:, 0] == task_ids.max(1)[0]) and torch.all(task_ids[:, 0] == task_ids.min(1)[0])
        task_ids = task_ids[:, 0].squeeze()
        
        results = torch.zeros_like(y_true)
        
        for task_id in task_ids.unique():
            task_mask = (task_ids == task_id)
            task_mu = mu[task_mask]
            task_sigma = sigma[task_mask]
            task_y_true = y_true[task_mask]
            
            # Compute pairwise differences
            y_diff = task_y_true.unsqueeze(0) - task_y_true.unsqueeze(1)  # Shape: (N, N)
            mu_diff = task_mu.unsqueeze(0) - task_mu.unsqueeze(1)        # Shape: (N, N)
            sigma_combined = torch.sqrt(task_sigma.unsqueeze(0)**2 + task_sigma.unsqueeze(1)**2)  # Shape: (N, N)

            # Mask to ignore self-comparisons and identical targets
            valid_mask = y_diff != 0  # Exclude y_diff == 0 (no ranking required)

            # Compute probabilistic scores
            prob = 0.5 * (1.0 + torch.erf(mu_diff / (sigma_combined * torch.sqrt(torch.tensor(2.0)))))

            # Loss for pairs where y_true[i] > y_true[j]
            positive_loss = -torch.log(prob + 1e-8) * (y_diff > 0)

            # Loss for pairs where y_true[i] < y_true[j]
            negative_loss = -torch.log(1.0 - prob + 1e-8) * (y_diff < 0)

            # Combine losses and apply valid mask
            total_loss = (positive_loss + negative_loss) * valid_mask
    
            loss = total_loss.sum(1) / valid_mask.sum(1)
            results[task_mask] = loss
            
        if self.reduction == "mean":
            return results.mean(0)

        return loss
    

class PairwiseRankingLoss(nn.Module):
    def __init__(self, bar_loss):
        """
        Initializes the PairwiseRankingLoss module.
        """
        super().__init__()
        self.criterion = bar_loss

    def forward(self, output, y_true, task_ids):
        """
        Compute the pairwise ranking loss based on incorrect orderings.

        Args:
            output (torch.Tensor): Predicted scores for the samples.
            target (torch.Tensor): Ground truth values for the samples.

        Returns:
            torch.Tensor: The computed pairwise ranking loss (fraction of incorrectly ordered pairs).
        """
        mu = self.criterion.mean(output)
        
        assert torch.all(task_ids[:, 0] == task_ids.max(1)[0]) and torch.all(task_ids[:, 0] == task_ids.min(1)[0])
        task_ids = task_ids[:, 0].squeeze()
        
        results = torch.zeros_like(y_true)
       
        for task_id in task_ids.unique():
            task_mask = (task_ids == task_id)
            task_mu = mu[task_mask]
            task_y_true = y_true[task_mask]

            # Compute pairwise differences
            y_diff = task_y_true.unsqueeze(0) - task_y_true.unsqueeze(1)  # Shape: (N, N)
            mu_diff = task_mu.unsqueeze(0) - task_mu.unsqueeze(1)        # Shape: (N, N)

            # Mask to ignore self-comparisons and identical targets
            valid_mask = y_diff != 0  # Exclude y_diff == 0 (no ranking required)

            # Loss for pairs where y_true[i] > y_true[j]
            positive_loss = (mu_diff <= 0).float() * (y_diff > 0)

            # Loss for pairs where y_true[i] < y_true[j]
            negative_loss = (mu_diff >= 0).float() * (y_diff < 0)

            # Combine losses and apply valid mask
            total_loss = (positive_loss + negative_loss) * valid_mask

            # Aggregate losses for the task
            task_loss = total_loss.sum() / valid_mask.sum()
            results[task_mask] = task_loss

        return results
    

class GaussianRankingLoss(nn.Module):
    def __init__(self, gaussian_weight=1.0, ranking_weight=1.0):
        super().__init__()
        self.gaussian_weight = gaussian_weight
        self.ranking_weight = ranking_weight
        self.gaussian_loss = nn.GaussianNLLLoss(full=True, reduction="none")
        self.ranking_loss = ProbabilisticRankingLoss(reduction="none")
        
    def forward(self, output, y_true, task_ids):
        mu = output[..., 0]
        var = output[..., 1].exp()
        
        gaussian = self.gaussian_loss(mu, y_true, var)
        ranking = self.ranking_loss(output, y_true, task_ids)
        
        return self.gaussian_weight * gaussian + self.ranking_weight * ranking


class BarRankingLoss(nn.Module):
    def __init__(self, bar_loss, bar_weight=1.0, ranking_weight=1.0):
        super().__init__()
        self.bar_weight = bar_weight
        self.ranking_weight = ranking_weight
        self.bar_loss = bar_loss
        self.ranking_loss = PairwiseRankingLoss(bar_loss)
        
    def forward(self, output, y_true, task_ids):
        bar = self.bar_loss(output, y_true)
        ranking = self.ranking_loss(output, y_true, task_ids)
        
        return self.bar_weight * bar + self.ranking_weight * ranking


    
class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = lambda num_classes: nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(num_classes)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution
    prob = ProbabilisticRankingLoss()

        
    
    def __repr__(self):
        return f"ProbabilisticRankingLoss"


def train(
    train_loader: prior.PriorDataLoader | callable,
    eval_loader: prior.PriorDataLoader | callable,
    criterion,
    encoder_generator,
    task_encoder_generator,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    dropout=0.0,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    seq_len=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    input_normalization=False,
    y_encoder_generator=None,
    pos_encoder_generator=None,
    decoder_dict={},
    train_extra_prior_kwargs_dict={},
    eval_extra_prior_kwargs_dict={},
    scheduler_generator=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_train_pos_gen=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    style_encoder_generator=None,
    epoch_callback=None,
    step_callback=None,
    continue_model=None,
    continue_scheduler=None,
    continue_optimizer=None,
    continue_epoch=None,
    initializer=None,
    initialize_with_model=None,
    train_mixed_precision=False,
    efficient_eval_masking=True,
    border_decoder=None,
    num_global_att_tokens=0,
    progress_bar=False,
    print_every=5,
    return_model=False,
    num_features=None,
    num_tasks=None,
    attn_type="axial",
    dirname=None,
    target_aware=False,
    task_embed_type="linear",
    save_val_results=False,
    **model_extra_args,
):
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    # print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)
    single_train_pos_gen = (
        single_train_pos_gen
        if callable(single_train_pos_gen)
        else lambda: single_train_pos_gen
    )
    def train_pos_seq_len_sampler(*args):
        single_train_pos = single_train_pos_gen(*args)
        return single_train_pos, seq_len

    # # LILY: what should this be
    # single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(
    #     seq_len - num_tasks, min_len=num_tasks
    # )
    def eval_pos_seq_len_sampler(*args):
        single_eval_pos = single_train_pos_gen(*args)
        return single_eval_pos, seq_len
    
    if inspect.isclass(train_loader) and issubclass(
        train_loader, prior.PriorDataLoader
    ):
        train_data_class = train_loader
    else:
        train_data_class = priors.utils.get_batch_to_dataloader(
            train_loader
        )
    
    if inspect.isclass(eval_loader) and issubclass(
        eval_loader, prior.PriorDataLoader
    ):
        eval_loader_class = eval_loader
    else:
        eval_loader_class = priors.utils.get_batch_to_dataloader(
            eval_loader
        )


    train_dl = train_data_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=train_pos_seq_len_sampler,
        seq_len_maximum=seq_len,
        device=device,
        **train_extra_prior_kwargs_dict,
    )
    
    eval_dl = eval_loader_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=seq_len,
        device=device,
        **eval_extra_prior_kwargs_dict,
    )

    test_batch: prior.Batch = train_dl.get_test_batch()
    style_def = test_batch.style
    # print(
    #     f"Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}"
    # )
    style_encoder = (
        style_encoder_generator(style_def.shape[1], emsize)
        if (style_def is not None)
        else None
    )
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(
        emsize, seq_len * 2
    )
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution)
        or "BarDistribution" in criterion.__class__.__name__
    ):  # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    elif isinstance(criterion, ProbabilisticRankingLoss):
        n_out = 2
    elif isinstance(criterion, GaussianRankingLoss):
        n_out = 2
    elif isinstance(criterion, BarRankingLoss):
        n_out = criterion.bar_loss.num_bars
    else:
        n_out = 1

    # border_decoder = None if border_decoder is None else border_decoder(emsize, criterion.num_bars + 1).to(device)

    if continue_model:
        model = continue_model
    else:
        decoder_dict = decoder_dict if decoder_dict else {"standard": (None, n_out)}

        decoder_once_dict = {}
        if test_batch.mean_prediction is not None:
            decoder_once_dict["mean_prediction"] = decoder_dict["standard"]

        if "axial" in attn_type:
            encoder = encoder_generator(1, emsize)
        else:
            if task_embed_type == "feature":
                encoder = encoder_generator(train_dl.num_features, train_dl.num_tasks, emsize)
            else:
                encoder = encoder_generator(train_dl.num_features, emsize)
            
        if task_embed_type == "linear":
            task_encoder = task_encoder_generator(1, emsize)
        elif task_embed_type == "onehot_linear":
            task_encoder = task_encoder_generator(train_dl.num_tasks, emsize)
        elif task_embed_type in ["feature", "task_attn", "task_attn_shared", "task_attn_opt", "hier", "hier_single"]:
            task_encoder = None
        else:
            raise ValueError("Invalid task_embed_type")
        
        # transformer_class = HierarchicalTransformerModel if "hier" in task_embed_type else TransformerModel
        model = TransformerModel(
            encoder=encoder,
            nhead=nhead,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            style_encoder=style_encoder,
            y_encoder=y_encoder_generator(1, emsize),
            input_normalization=input_normalization,
            pos_encoder=pos_encoder,
            decoder_dict=decoder_dict,
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            decoder_once_dict=decoder_once_dict,
            num_global_att_tokens=num_global_att_tokens,
            num_features=num_features,
            num_tasks=num_tasks,
            attn_type=attn_type,
            task_encoder=task_encoder,
            task_embed_type=task_embed_type,
            include_global=(not "single" in task_embed_type),
            hierarchical=("hier" in task_embed_type),
            **model_extra_args,
        )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(), initialize_with_model.state_dict().items()
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    # learning rate
    if continue_optimizer:
        optimizer = continue_optimizer
        scheduler = continue_scheduler
    else:
        if lr is None:
            lr = get_openai_lr(model)
            print(f"Using OpenAI max lr of {lr}.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = scheduler_generator(
            optimizer, warmup_epochs, epochs if epochs is not None else 100
        )  # when training for fixed time lr schedule takes 100 steps

    if return_model:
        return model, optimizer, scheduler
    
    wandb.watch(model, log="all")
    model.to(device)
    # optimizer is on device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=test_batch.mean_prediction is not None,
        )
        train_dl.model = (
            model.module
        )  # use local model, should not use multi-gpu functionality..
        eval_dl.model = model.module
    else:
        train_dl.model = model
        eval_dl.model = model

    scaler = torch.amp.GradScaler("cuda") if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(train_dl)

    best_val_score = 1e9
    best_model = None
    best_train_score = 1e9
    best_model_train = None

    def run_epoch(train_mode=True, dl=train_dl, epoch=0):
        if train_mode:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        
        n_batches = len(dl)
        if not train_mode:
            n_batches = min(512, n_batches)
            
        if train_mode:
            assert (
                len(dl) % aggregate_k_gradients == 0
            ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
            tqdm_iter = (
                tqdm(range(n_batches), desc="Training Epoch %d" % epoch)
                if rank == 0 and progress_bar
                else None
            )  # , disable=not verbose
        else:
            tqdm_iter = (
                tqdm(range(n_batches), desc="Eval Epoch %d" % epoch)
                if rank == 0 and progress_bar
                else None
            )

        for batch, full_data in enumerate(dl):
            if batch > n_batches:
                break
            
            data = (
                full_data.style.to(device) if full_data.style is not None else None,
                full_data.x.to(device),
                full_data.y.to(device),
                full_data.task_id.to(device) if full_data.task_id is not None else None,
            )
            targets = full_data.target_y.to(device)
            single_eval_pos = full_data.single_eval_pos
            # print("epoch", epoch, "single_eval_pos", single_eval_pos)

            def get_metrics():
                return (
                    total_loss / n_batches,
                    (
                        total_positional_losses / total_positional_losses_recorded
                    ).tolist(),
                    time_to_get_batch,
                    forward_time,
                    step_time,
                    nan_steps.cpu().item() / (batch + 1),
                    ignore_steps.cpu().item() / (batch + 1),
                    n_batches * batch_size,
                )

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (
                batch % aggregate_k_gradients == aggregate_k_gradients - 1
            ):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                try:
                    metrics_to_log = {}
                    with torch.amp.autocast("cuda", enabled=scaler is not None):
                        # If style is set to None, it should not be transferred to device
                        if train_mode:
                            out = model(
                                tuple(
                                    e.to(device) if torch.is_tensor(e) else e for e in data
                                ),
                                single_eval_pos=single_eval_pos,
                                only_return_standard_out=False,
                            )
                        else:
                            with torch.no_grad():
                                out = model(
                                    tuple(
                                        e.to(device)
                                        if torch.is_tensor(e)
                                        else e
                                        for e in data
                                    ),
                                    single_eval_pos=single_eval_pos,
                                    only_return_standard_out=False,
                                )

                        # this handling is for training old models only, this can be deleted soon(ish)
                        # to only support models that return a tuple of dicts
                        out, output_once = (
                            out if isinstance(out, tuple) else (out, None)
                        )
                        output = out["standard"] if isinstance(out, dict) else out

                        forward_time = time.time() - before_forward

                        # only evaluate on target_task
                        # x is seq, batch, feature

                        if single_eval_pos is not None:
                            targets = targets[single_eval_pos:]
                            target_task_ids = full_data.task_id[single_eval_pos:]
                        else:
                            target_task_ids = full_data.task_id
                        
                        if len(targets.shape) == len(output.shape):
                            # this implies the prior uses a trailing 1 dimesnion
                            # below we assume this not to be the case
                            targets = targets.squeeze(-1)
                        assert targets.shape == output.shape[:-1], (
                            f"Target shape {targets.shape} "
                            "does not match output shape {output.shape}"
                        )
                        if isinstance(criterion, nn.GaussianNLLLoss):
                            assert (
                                output.shape[-1] == 2
                            ), "need to write a little bit of code to handle multiple regression targets at once"

                            mean_pred = output[..., 0]
                            var_pred = output[..., 1].exp()
                            losses = criterion(mean_pred, targets, var=var_pred)
                        elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                            targets[torch.isnan(targets)] = -100
                            losses = criterion(output.flatten(), targets.flatten())
                        elif isinstance(criterion, nn.CrossEntropyLoss):
                            targets[torch.isnan(targets)] = -100
                            print(f"{targets.min()=}, {targets.max()=}")
                            losses = criterion(
                                output.reshape(-1, n_out), targets.long().flatten()
                            )
                        elif isinstance(criterion, ProbabilisticRankingLoss):
                            losses = criterion(output, targets, target_task_ids)
                        elif isinstance(criterion, GaussianRankingLoss):
                            losses = criterion(output, targets, target_task_ids)
                        elif isinstance(criterion, BarRankingLoss):
                            losses = criterion(output, targets, target_task_ids)
                        elif border_decoder is not None:

                            def apply_batch_wise_criterion(i):
                                output_, targets_, borders_ = (
                                    output_adaptive[:, i],
                                    targets[:, i],
                                    borders[i],
                                )
                                criterion_ = get_custom_bar_dist(
                                    borders_, criterion
                                ).to(device)
                                return criterion_(output_, targets_)

                            output_adaptive, borders = (
                                out["adaptive_bar"],
                                output_once["borders"],
                            )
                            losses_adaptive_bar = torch.stack(
                                [
                                    apply_batch_wise_criterion(i)
                                    for i in range(output_adaptive.shape[1])
                                ],
                                1,
                            )
                            losses_fixed_bar = criterion(output, targets)
                            losses = (losses_adaptive_bar + losses_fixed_bar) / 2

                            metrics_to_log = {
                                **metrics_to_log,
                                **{
                                    "loss_fixed_bar": losses_fixed_bar.mean()
                                    .cpu()
                                    .detach()
                                    .item(),
                                    "loss_adaptive_bar": losses_adaptive_bar.mean()
                                    .cpu()
                                    .detach()
                                    .item(),
                                },
                            }
                        elif (
                            isinstance(criterion, BarDistribution)
                            and full_data.mean_prediction
                        ):
                            assert "mean_prediction" in output_once
                            utils.print_once("Using mean prediction for loss")
                            losses = criterion(
                                output,
                                targets,
                                mean_prediction_logits=output_once["mean_prediction"],
                            )
                            # the mean pred loss appears as the last per sequence
                        else:
                            losses = criterion(output, targets)
                        losses = losses.flatten()
                        loss, nan_share = utils.torch_nanmean(
                            losses.mean(0, keepdim=True), return_nanshare=True
                        )
                        loss_scaled = loss / aggregate_k_gradients

                    if scaler:
                        loss_scaled = scaler.scale(loss_scaled)
                        
                    if train_mode:
                        loss_scaled.backward()

                        if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                            if scaler:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    if not torch.isnan(loss):
                        total_loss += loss.cpu().detach().item()
                        total_positional_losses += (
                            losses.mean(1).cpu().detach()
                            if single_eval_pos is None
                            else nn.functional.one_hot(
                                torch.tensor(single_eval_pos), seq_len
                            )
                            * utils.torch_nanmean(
                                losses[: seq_len - single_eval_pos].mean(0)
                            )
                            .cpu()
                            .detach()
                        )

                        total_positional_losses_recorded += (
                            torch.ones(seq_len)
                            if single_eval_pos is None
                            else nn.functional.one_hot(
                                torch.tensor(single_eval_pos), seq_len
                            )
                        )

                        metrics_to_log = {
                            **metrics_to_log,
                            **{f"train_loss": loss, "single_eval_pos": single_eval_pos},
                        }
                        # wandb.log(metrics_to_log)
                        if step_callback is not None and rank == 0:
                            step_callback(metrics_to_log)
                        nan_steps += nan_share
                        ignore_steps += (targets == -100).float().mean()
                except Exception as e:
                    print("Invalid step encountered, skipping...")
                    print(e)
                    raise (e)

            # total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share = get_metrics()
            if tqdm_iter:
                tqdm_iter.set_postfix(
                    {
                        "loss": total_loss / (batch + 1),
                        "data": time_to_get_batch,
                        "step": step_time,
                    }
                )

            before_get_batch = time.time()
        metrics = get_metrics()
        return metrics

    total_loss = float("inf")
    total_positional_losses = float("inf")
    # Initially test the epoch callback function
    if epoch_callback is not None and rank == 0:
        epoch_callback(model, 1, data_loader=train_dl, scheduler=scheduler)
    # start_epochs = 1 if continue_epoch is None else continue_epoch + 1
    start_epochs = 0 if continue_epoch is None else continue_epoch
    
    # for epoch in range(start_epochs, epochs + 1) if epochs is not None else itertools.count(1):
    epoch = start_epochs
    while True:
        epoch += 1
        if epochs is not None and epoch > epochs:
            break
        epoch_start_time = time.time()
        try:
            (
                total_loss,
                total_positional_losses,
                time_to_get_batch,
                forward_time,
                step_time,
                nan_share,
                ignore_share,
                data_size,
            ) = run_epoch(train_mode=True, dl=train_dl, epoch=epoch)
            
            wandb.log({
                "epoch": epoch, 
                "data_size": data_size * epoch,
                "train_loss": total_loss,
            })
            # save checkpoint
            torch.save(model.state_dict(), os.path.join(dirname, "final_model.pth"))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "total_loss": total_loss,
                "seed": torch.initial_seed(),
            }, os.path.join(dirname, "checkpoint.pth")
            )
            with open(os.path.join(dirname, 'result.json'), 'w') as f:
                json.dump({"epoch": epoch, "train_loss": total_loss}, f)
            
            if total_loss < best_train_score:
                best_train_score = total_loss
                torch.save(model.state_dict(), os.path.join(dirname, "best_train_model.pth"))
            
        except Exception as e:
            print("Invalid epoch encountered, skipping...")
            print(e)
            raise (e)
        # if hasattr(dl, "validate") and epoch % validation_period == 0:
        if epoch % validation_period == 0:
            with torch.no_grad():
                val_score = run_epoch(train_mode=False, dl=eval_dl, epoch=epoch)[0]
                wandb.log({
                    "eval_loss": val_score,
                }, step=wandb.run.step)
                if val_score < best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(dirname, "best_model.pth"))
                if save_val_results:
                    torch.save(model.state_dict(), os.path.join(dirname, f"model_{epoch}.pth"))

        else:
            val_score = None

        if verbose and epoch % print_every == 0:
            if val_score is not None:
                text = (
                    "-" * 89
                    + "\n"
                    + f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | train loss {total_loss:5.2f} | val loss {val_score:5.2f} |\n"
                    # f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
                    f" forward time {forward_time:5.2f}"
                    # f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    # + (f" val score {val_score:5.2f}" if val_score is not None else "")
                    + "\n"
                    + "-" * 89
                    + "\n"
                )
            else:
                text = (
                    "-" * 89
                    + "\n"
                    + f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | train loss {total_loss:5.2f} |\n"
                    # f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
                    f" forward time {forward_time:5.2f}"
                    # f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    # + (f" val score {val_score:5.2f}" if val_score is not None else "")
                    + "\n"
                    + "-" * 89
                    + "\n"
                )
            logging.info(text)
            # print("-" * 89)
            # print(
            #     f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | "
            #     # f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
            #     f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
            #     f" forward time {forward_time:5.2f}"
            #     # f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
            #     + (f" val score {val_score:5.2f}" if val_score is not None else "")
            # )
            # print("-" * 89)

        # stepping with wallclock time based scheduler
        if epoch_callback is not None and rank == 0:
            epoch_callback(model, epoch, data_loader=train_dl, scheduler=scheduler)
        scheduler.step()

    if rank == 0:  # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            train_dl = None
        return (
            total_loss,
            total_positional_losses,
            model.to("cpu"),
            train_dl,
            best_val_score,
            best_model,
        )


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text