#!/usr/bin/env python3

import torch
from gen_batch import (
    combine_batch,
    get_mtgp_batch,
    get_pd1_surrogate_batch_fn,
    get_pd1_eval_batch_fn,
    get_trios_batch,
    get_hpobench_batch_fn,
    target_aware_mtgp_batch,
    gen_mtgp_equal_eval_batch,
    gen_mtgp_low_rank,
)
from paper_figures.joint_data import get_joint_batch
from gen_axial_batch import axial_train_batch, multitask_line_batch
from gen_task_batch import task_invariant_batch, task_corr_batch
from PFNs.pfns import (
    bar_distribution,
    encoders,
    priors,
    utils,
)
from PFNs.pfns.train import (
    train as train_pfn,
    ProbabilisticRankingLoss,
    PairwiseRankingLoss,
    GaussianRankingLoss,
    BarRankingLoss,
)
import sys
import logging
import os
import datetime
import json
import argparse
import traceback
import wandb
import numpy as np

default_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train_from_checkpoint(checkpoint_folder, wandb_mode="online", pty=False, epochs=None, seed=None, lr=None, **kwargs):
    args_path = checkpoint_folder + '/args.json'
    with open(args_path, 'r') as f:
        args = json.load(f)
    args["wandb_mode"] = wandb_mode
    args["checkpoint_folder"] = checkpoint_folder
    args["pty"] = pty
    
    if epochs is None or seed is None or lr is None:
        print("Using original epochs, seed, and lr")
        continue_scheduler = True
        
        checkpoint_path = checkpoint_folder + '/checkpoint.pth'
        checkpoint = torch.load(checkpoint_path, map_location=default_device)
        args["start_epoch"] = checkpoint["epoch"]
    else:
        print(f"Restarting with epochs={epochs}, seed={seed}, lr={lr}")
        continue_scheduler = False
        
        args["epochs"] = epochs
        args["seed"] = seed
        args["lr"] = lr
        
    for key, val in kwargs.items():
        if val is not None:
            args[key] = val
    
    random_num = np.random.randint(100000)
    dirname = create_experiment_dir(argparse.Namespace(**args), random_num, continued=True)
    args["dirname"] = dirname
    
    wandb.init(
        project="mtpfn",
        config=args,
        mode=wandb_mode,
    )
    
    # make symlink from wandb name to dirname
    if wandb_mode != "disabled":
        wandb_name = wandb.run.name
        # wandb_dir = "/home/yl9959/mtpfn/wandb_links"
        wandb_dir = args["wandb_dir"]
        os.symlink(dirname, os.path.join(wandb_dir, wandb_name))
    
    if pty:
        return train(**args, progress_bar=True, continue_model=True, continue_scheduler=continue_scheduler) 
    else:
        # redirect stdout and stderr
        sys.stdout = open(os.path.join(dirname, 'stdout.txt'), 'w')
        sys.stderr = open(os.path.join(dirname, 'stderr.txt'), 'w')
        return train(**args, continue_model=True, continue_scheduler=continue_scheduler)


def train_continue(continue_scheduler, checkpoint_folder, config, dirname):
    checkpoint_path = checkpoint_folder + '/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=default_device)
    
    model, optimizer, scheduler = train(**config, return_model=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    config["continue_model"] = model
    
    if continue_scheduler:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        
        config["continue_optimizer"] = optimizer
        config["continue_scheduler"] = scheduler
        config["continue_epoch"] = epoch
    return train_pfn_wrapper(config, dirname)


def train_pfn_wrapper(config, dirname):
    train_result = train_pfn(**config, dirname=dirname)
    total_loss, total_positional_losses, final_model, dl, best_val_score, best_model = (
        train_result
    )

    if final_model:
        final_state = final_model.state_dict()
    if best_model:
        best_state = best_model.state_dict()

    results = {
        "final_loss": total_loss,
        "final_epoch": config["epochs"],
    }
    torch.save(final_state, os.path.join(dirname, "final_model.pth"))
    
    if best_model:
        torch.save(best_state, os.path.join(dirname, "best_model.pth"))
        results["best_loss"] = best_val_score
        
    with open(os.path.join(dirname ,'result.json'), 'w') as f:
        json.dump(results, f, indent=4)


def get_batch_gen(prior_type, **kwargs):
    hypers = {}
    
    if prior_type == "mtgp":
        get_batch_fn = gen_mtgp_equal_eval_batch
        hypers["corr_init"] = kwargs["corr_init"]
        hypers["uncorr_prob"] = kwargs["uncorr_prob"]
    elif prior_type == "mtgp_1_uncorr":
        get_batch_fn = gen_mtgp_equal_eval_batch
        hypers["num_uncorr_tasks"] = 1
    elif prior_type == "mtgp_4_1_uncorr":
        get_batch_fn = gen_mtgp_equal_eval_batch
        hypers["num_uncorr_tasks"] = 1
        hypers["num_tasks"] = 4
    elif prior_type == "mtgp_4_1_4_uncorr":
        get_batch_fn = gen_mtgp_equal_eval_batch
        hypers["num_uncorr_tasks"] = 1
        hypers["num_tasks"] = 4
        hypers["num_features"] = 4
    elif prior_type == "toy_axial":
        get_batch_fn = axial_train_batch
    elif prior_type == "toy_multitask":
        get_batch_fn = multitask_line_batch
    elif prior_type == "toy_task_invariant":
        get_batch_fn = task_invariant_batch
    elif prior_type == "toy_corr_multitask":
        get_batch_fn = task_corr_batch
    elif prior_type == "pd1":
        get_batch_fn = get_pd1_surrogate_batch_fn(**kwargs)
    elif prior_type == "pd1_eval":
        get_batch_fn = get_pd1_eval_batch_fn(**kwargs)
    elif prior_type == "trio":
        get_batch_fn = get_trios_batch
    elif prior_type == "hpobench_lr":
        get_batch_fn = get_hpobench_batch_fn("lr", train=True)
    elif prior_type == "hpobench_lr_eval":
        get_batch_fn = get_hpobench_batch_fn("lr", train=False)
    elif prior_type == "hpobench_svm":
        get_batch_fn = get_hpobench_batch_fn("svm", train=True)
    elif prior_type == "hpobench_svm_eval":
        get_batch_fn = get_hpobench_batch_fn("svm", train=False)
    elif prior_type == "target_aware":
        get_batch_fn = target_aware_mtgp_batch
        hypers["corr_init"] = kwargs["corr_init"]
        hypers["uncorr_prob"] = kwargs["uncorr_prob"]
    elif prior_type == "target_aware_4_1_4_uncorr":
        get_batch_fn = target_aware_mtgp_batch
        hypers["num_uncorr_tasks"] = 1
        hypers["num_tasks"] = 4
        hypers["num_features"] = 4
    elif prior_type == "target_aware_4_1_2_uncorr":
        get_batch_fn = target_aware_mtgp_batch
        hypers["num_uncorr_tasks"] = 1
        hypers["num_tasks"] = 4
        hypers["num_features"] = 2
    elif prior_type == "low_rank":
        get_batch_fn = gen_mtgp_low_rank
        hypers["decay_alpha"] = kwargs["decay_alpha"]
        hypers["max_rank"] = kwargs["max_rank"]
    else:
        raise ValueError("prior_type is invalid")

    return get_batch_fn, hypers


def get_batch_gen_and_prior_dl(prior_type, **kwargs):
    if prior_type == "combine":
        functions = []
        hypers = {}
        for prior_type in kwargs["prior_types"]:
            get_batch_fn, prior_hypers = get_batch_gen(prior_type, **kwargs)
            functions.append(get_batch_fn)
            hypers = {**hypers, **prior_hypers}
        weights = torch.tensor(kwargs["prior_weights"])
        get_batch_fn = combine_batch(functions, weights)
    else:
        get_batch_fn, hypers = get_batch_gen(prior_type, **kwargs)

    return priors.get_batch_sequence(
        get_batch_fn,
        priors.utils.set_properties_get_batch,
    ), hypers


def train(
    attn_type="axial",
    num_features=40,
    num_tasks=5,
    prior_type="mtgp",
    lr=0.0001,
    epochs=50,
    seq_len=200,
    single_train_pos_gen_type="uniform",
    sample_num_tasks=True,
    save_to_manifold=True,
    corr_init=0.8,
    sample_num_features=True,
    scheduler_type="cosine",
    lengthscale=None,
    criterion_type="gaussian",
    uncorr_prob=0.0,
    emsize=32,
    task_embed_type="linear",
    same_tasks_across_batch=True,
    full_precision=False,
    return_model=False,  # not included in config
    device=default_device,
    dirname=None,
    continue_model=False,
    continue_scheduler=False,
    **kwargs,
):
    if kwargs.get("eval_type", None) is None:
        kwargs["eval_type"] = prior_type
        
    train_config = {
        "attn_type": attn_type,
        "prior_type": prior_type,
        "num_features": num_features,
        "num_tasks": num_tasks,
        "lr": lr,
        "epochs": epochs,
        "seq_len": seq_len,
        "single_train_pos_gen_type": single_train_pos_gen_type,
        "sample_num_tasks": sample_num_tasks,
        "save_to_manifold": save_to_manifold,
        "corr_init": corr_init,
        "sample_num_features": sample_num_features,
        "scheduler_type": scheduler_type,
        "lengthscale": lengthscale,
        "criterion_type": criterion_type,
        "uncorr_prob": uncorr_prob,
        "emsize": emsize,
        "task_embed_type": task_embed_type,
        **kwargs,
    }

    # get batch function
    get_train_batch_fn, train_hypers = get_batch_gen_and_prior_dl(
        max_features=num_features,
        min_num_tasks=num_tasks,
        **train_config,
    )
    eval_config = train_config.copy()
    eval_config["prior_type"] = train_config["eval_type"]
    get_eval_batch_fn, eval_hypers = get_batch_gen_and_prior_dl(
        max_features=num_features,
        min_num_tasks=num_tasks,
        **eval_config,
    )

    batch_hypers = {
        "num_tasks": num_tasks,
        # "sample_num_tasks": sample_num_tasks,
        # "sample_num_features": sample_num_features,
        "lengthscale": lengthscale,
        "same_tasks_across_batch": same_tasks_across_batch,
        "target_only_loss": kwargs.get("target_only_loss", False),
    }
    # combine train and batch hypers
    train_batch_fn_hypers = {**train_hypers, **batch_hypers, "sample_num_tasks": sample_num_tasks, "sample_num_features": sample_num_features, "sample_num_tasks_curriculum": kwargs.get("sample_num_tasks_curriculum", False)}
    eval_batch_fn_hypers = {**eval_hypers, **batch_hypers, "sample_num_tasks": False, "sample_num_features": False}

    if "bar" in criterion_type:
        if return_model or continue_model:
            n = 100
        else:
            n = 100000
        ys = get_train_batch_fn(
            batch_size=n,
            seq_len=20,
            num_features=num_features,
            hyperparameters=train_batch_fn_hypers,
            device=device,
        ).target_y.to(device)
        criterion = bar_distribution.FullSupportBarDistribution(
            bar_distribution.get_bucket_limits(num_outputs=1000, ys=ys)
        )
        if criterion_type == "bar_ranking":
            criterion = BarRankingLoss(criterion)
    elif criterion_type == "gaussian":
        criterion = torch.nn.GaussianNLLLoss(reduction="none", full=True)
    elif criterion_type == "prob_ranking":
        criterion = ProbabilisticRankingLoss(reduction="none")
    elif criterion_type == "gaussian_ranking":
        criterion = GaussianRankingLoss()
    # elif criterion_type == "ranking":
    #     criterion = PairwiseRankingLoss()
    else:
        raise ValueError("criterion_type must be one of ['bar', 'gaussian', 'prob']")

    if "axial" in attn_type:
        encoder_generator = encoders.get_axial_normalized_uniform_encoder()
    elif attn_type == "standard":
        if task_embed_type == "feature":
            encoder_generator = encoders.get_normalized_uniform_multitask_encoder(
                encoders.get_variable_num_features_multitask_encoder(
                    encoders.Linear
                )
            )
        else:
            encoder_generator = encoders.get_normalized_uniform_encoder(
                encoders.get_variable_num_features_encoder(encoders.Linear)
            )
        
    if task_embed_type == "linear":
        task_encoder_generator = encoders.Linear
    elif task_embed_type == "onehot_linear":
        task_encoder_generator = encoders.Linear
    elif task_embed_type == "self_attn":
        task_encoder_generator = encoders.get_self_attention_task_encoder()
    elif task_embed_type == "feature":
        task_encoder_generator = None
    elif task_embed_type == "task_attn":
        task_encoder_generator = None
    elif task_embed_type == "task_attn_shared":
        task_encoder_generator = None
    elif task_embed_type == "task_attn_opt":
        task_encoder_generator = None
    elif task_embed_type == "hier" or task_embed_type == "hier_single":
        task_encoder_generator = None
    else:
        raise ValueError("task_embed_type must be one of ['linear', 'onehot_linear', 'self_attn', 'feature', 'task_attn', 'task_attn_shared']")
    
    if single_train_pos_gen_type == "decay":
        single_train_pos_gen = utils.get_exponential_decay_single_eval_pos_sampler(
            seq_len - num_tasks, min_len=num_tasks
        )
    elif single_train_pos_gen_type == "uniform":
        single_train_pos_gen = utils.get_uniform_single_eval_pos_sampler(
            seq_len - num_tasks - kwargs.get("min_eval_len", 0), min_len=num_tasks
        )
    elif single_train_pos_gen_type == "uniform_large":
        single_train_pos_gen = utils.get_uniform_single_eval_pos_sampler(
            seq_len - num_tasks - 100, min_len=num_tasks
        )
    elif single_train_pos_gen_type == "curriculum":
        single_train_pos_gen = utils.get_curriculum_single_eval_pos_sampler(
            seq_len - num_tasks, min_len=num_tasks
        )
    else:
        raise ValueError("single_train_pos_gen_type must be one of ['uniform', 'decay']")

    if scheduler_type == "cosine":
        scheduler = utils.get_cosine_schedule_with_warmup
    elif scheduler_type == "restart":
        scheduler = utils.get_restarting_cosine_schedule_with_warmup
    elif scheduler_type == "restart_slow":
        scheduler = utils.get_slow_restarting_cosine_schedule_with_warmup
    elif scheduler_type == "curriculum":
        scheduler = utils.get_curriculum_cosine_schedule_with_warmup
    else: 
        raise ValueError(
            "scheduler_type must be one of ['cosine', 'restart', 'restart_slow']"
        )
        
    config = {
        "train_loader": get_train_batch_fn,
        "eval_loader": get_eval_batch_fn,
        "criterion": criterion,
        "encoder_generator": encoder_generator,
        "task_encoder_generator": task_encoder_generator,
        "single_train_pos_gen": single_train_pos_gen,
        "scheduler": scheduler,
        "emsize": emsize,
        "nhead": 4,
        "warmup_epochs": 5,
        "y_encoder_generator": encoders.Linear,
        "batch_size": 128,
        "train_extra_prior_kwargs_dict": {
            "num_features": num_features,
            "hyperparameters": train_batch_fn_hypers,
        },
        "eval_extra_prior_kwargs_dict": {
            "num_features": eval_batch_fn_hypers.get("num_features", num_features),
            "hyperparameters": eval_batch_fn_hypers,
        },
        "epochs": 50,
        "lr": 0.0001,
        "seq_len": 200,
        "aggregate_k_gradients": 2,
        "nhid": 1024,
        "steps_per_epoch": 1024,
        "weight_decay": 0.0,
        "train_mixed_precision": not full_precision,
        "efficient_eval_masking": True,
        "nlayers": kwargs["n_layers"] if "n_layers" in kwargs else 6,
        "print_every": 5,
        "validation_period": 5,
        "meta_tokens": kwargs["meta_tokens"] if "meta_tokens" in kwargs else 1,
        "same_tasks_across_batch": same_tasks_across_batch,
    }
    
    for key, val in train_config.items():
        # if key not in config:
        #     print(f"Adding {key} to config")
        config[key] = val

    if return_model:
        config["return_model"] = True
        return train_pfn(**config)
    
    # if "checkpoint_folder" in kwargs:
    #     return train_continue(kwargs["checkpoint_folder"], config, dirname)
    if continue_model:
        return train_continue(continue_scheduler, kwargs["checkpoint_folder"], config, dirname)

    return train_pfn_wrapper(config, dirname)




def create_experiment_dir(args, random_num, continued=False):
    dirname = args.output_dir

    # name of experiment
    exp_name = "prior_%s" % args.prior_type
    exp_name += "__features_%d__tasks_%d" % (args.num_features, args.num_tasks)
    if args.epochs is not None:
        exp_name += "__epochs_%d" % args.epochs
    else:
        exp_name += "__epochs_None"
    exp_name += "__seqlen_%d" % args.seq_len
    exp_name += "__attn_%s" % args.attn_type
    exp_name += "__task_%s" % args.task_embed_type
    exp_name += "__seed_%d" % args.seed

    if args.permute_tasks:
        exp_name += "__permute"
    
    if args.prior_type == "combine":
        for prior_type, weight in zip(args.prior_types, args.prior_weights):
            weight_string = str(weight).replace(".", "_")
            exp_name += "__%s_%s" % (prior_type, weight_string)
    
    if args.exp_name:
        exp_name = args.exp_name + "__" + exp_name
        
    if args.smoke:
        exp_name = "smoke_" + exp_name
    
    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    dirname = os.path.join(args.output_dir, current_time + "__" + exp_name)
    if continued:
        dirname += "__continued"
    dirname += "__%d" % random_num

    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    # save arguments as json
    with open(os.path.join(dirname ,'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # # redirect stdout and stderr
    # if args.redirect:
    #     sys.stdout = open(os.path.join(dirname, 'stdout.txt'), 'w')
    #     sys.stderr = open(os.path.join(dirname, 'stderr.txt'), 'w')

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s: %(message)s')
    logging.info("Experiment directory: %s\n\n" % dirname)
    
    return dirname


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    # properties of model
    argparser.add_argument('--num_features', type=int, default=7)
    argparser.add_argument('--num_tasks', type=int, default=4)
    argparser.add_argument('--sample_num_tasks', action='store_true', default=False)
    argparser.add_argument('--sample_num_tasks_curriculum', action='store_true', default=False)
    argparser.add_argument('--sample_num_features', action='store_true', default=False)
    
    # model architecture
    argparser.add_argument('--attn_type', type=str, default='standard', help='axial, axial_parallel, or standard')
    argparser.add_argument('--task_embed_type', type=str, default='hier', help='{linear, onehot_linear, feature, task_attn, task_attn_shared}')
    # hierarchical
    argparser.add_argument('--meta_tokens', type=int, default=1)
    
    # prior properties
    argparser.add_argument('--prior_type', type=str, default='mtgp')
    argparser.add_argument('--target_only_loss', action='store_true', default=False)
    argparser.add_argument('--same_tasks_across_batch', action='store_true', default=False)
    # for low-rank
    argparser.add_argument('--max_rank', type=int, default=5)
    argparser.add_argument('--decay_alpha', type=float, default=0.2)
    # for gp-based
    argparser.add_argument('--lengthscale', type=float, default=None)
    # for unrelated
    argparser.add_argument('--uncorr_prob', type=float, default=0.0)
    # for mtgp-bias
    argparser.add_argument('--corr_init', type=float, default=None)
    # for combine prior
    argparser.add_argument('--prior_types', nargs='+', default=None)
    argparser.add_argument('--prior_weights', nargs='+', type=float, default=None)
    # toy
    argparser.add_argument('--permute_tasks', action='store_true', default=False)
    
    # eval properties
    argparser.add_argument('--eval_type', type=str, default=None)
    
    # properties of training
    argparser.add_argument('--seq_len', type=int, default=400)
    argparser.add_argument('--single_train_pos_gen_type', type=str, default='uniform')
    argparser.add_argument('--min_eval_len', type=int, default=200)
    argparser.add_argument('--criterion_type', type=str, default='bar')
    
    # training hyperparameters
    argparser.add_argument('--lr', type=float, default=0.0001)
    argparser.add_argument('--epochs', type=int, default=None)
    argparser.add_argument('--scheduler_type', type=str, default='cosine')
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--steps_per_epoch', type=int, default=1024)
    argparser.add_argument('--emsize', type=int, default=512)
    argparser.add_argument('--nhid', type=int, default=2048)
    argparser.add_argument('--n_layers', type=int, default=24)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--full_precision', action='store_true', default=False)
    
    argparser.add_argument('--target_aware', action='store_true', default=False)
    argparser.add_argument('--global_with_target_points', action='store_true', default=False)
    argparser.add_argument('--local_with_target_points', action='store_true', default=False)
    
    # identify run
    if os.path.exists("/home/yl9959/mtpfn/ckpt"):
        output_dir = "/home/yl9959/mtpfn/ckpt"
        wandb_dir = "/home/yl9959/mtpfn/wandb_links"
    else:
        output_dir = "/home/lily_l/private_multitask_pfn/ckpt"
        wandb_dir = "/home/lily_l/private_multitask_pfn/wandb_links"
    argparser.add_argument('--output_dir', type=str, default=output_dir)
    argparser.add_argument('--wandb_dir', type=str, default=wandb_dir)
    
    argparser.add_argument('--exp_name', type=str, default=None)
    argparser.add_argument('--smoke', action='store_true', default=False)
    argparser.add_argument('--pty', action='store_true', default=False)
    argparser.add_argument('--disable_wandb', action='store_true', default=False)
    args = argparser.parse_args()
    
    
    random_num = np.random.randint(100000)
    dirname = create_experiment_dir(args, random_num)
    config = vars(args)
    config["dirname"] = dirname
    # if not args.pty and not args.smoke:
    #     # redirect stdout and stderr
    #     sys.stdout = open(os.path.join(dirname, 'stdout.txt'), 'w')
    #     sys.stderr = open(os.path.join(dirname, 'stderr.txt'), 'w')
    
    if args.disable_wandb or args.smoke:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
        
    wandb.init(
        project="mtpfn",
        config=config,
        mode=wandb_mode,
    )
    
    # make symlink from wandb name to dirname
    if wandb_mode != "disabled":
        wandb_name = wandb.run.name
        wandb_dir = args.wandb_dir
        os.symlink(dirname, os.path.join(wandb_dir, wandb_name))
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        if args.smoke:
            args.epochs = 5
            args.steps_per_epoch = 32
            train(**config, progress_bar=True)
        else:
            if args.pty:
                train(**config, progress_bar=True)
            else:
                train(**config)
    except Exception as e:
        print("EXCEPTION")
        # rename directory to indicate failure
        new_dirname = dirname + "_failed"
        os.rename(dirname, new_dirname)
        
        if wandb_mode != "disabled":
            # delete original symlink
            os.remove(os.path.join(wandb_dir, wandb_name))
            # create new symlink
            os.symlink(new_dirname, os.path.join(wandb_dir, wandb_name))
            
        traceback.print_exc()  # This will print the full traceback
        # Re-raise the exception with its original stack trace
        raise
    
    # os.rename(dirname, dirname + "_completed")