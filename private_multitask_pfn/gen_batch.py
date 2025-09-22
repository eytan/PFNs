#!/usr/bin/env python3
import gpytorch

import torch
from PFNs.pfns.priors import Batch
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from linear_operator.operators import InterpolatedLinearOperator
import json
import pickle
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.exceptions import ModelFittingError
from gpytorch.mlls import ExactMarginalLogLikelihood
from eval_hpobench import get_torch_format_hpobench
import os

default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


class IndexKernelFixed(Kernel):
    def __init__(
        self,
        covar_matrix,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.covar_matrix = covar_matrix

    def forward(self, i1, i2, **params):
        i1, i2 = i1.long(), i2.long()
        covar_matrix = self.covar_matrix
        batch_shape = torch.broadcast_shapes(
            i1.shape[:-2], i2.shape[:-2], self.batch_shape
        )

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res


def sample_task_ids(batch_size: int, seq_len: int, num_tasks: int, device: str, same_tasks_across_batch=False):
    # split data into target and source tasks with Dirichlet / Categorical sampling
    alphas = torch.ones(num_tasks).to(device) * 2.0
    # alphas[0] = 1.0
    
    
    # B x N
    if same_tasks_across_batch:
        probabilities = torch.distributions.Dirichlet(alphas).sample()
        task_id = torch.distributions.Categorical(probabilities).sample((seq_len,)).T.to(device)
        task_id = task_id.unsqueeze(0).repeat(batch_size, 1)
    else:
        probabilities = torch.distributions.Dirichlet(alphas).sample((batch_size,))
        task_id = (
            torch.distributions.Categorical(probabilities).sample((seq_len,)).T.to(device)
        )
    task_id[:, :num_tasks] = torch.arange(num_tasks).repeat(batch_size, 1).to(device)
    task_id[:, -num_tasks:] = torch.arange(num_tasks).repeat(batch_size, 1).to(device)

    return task_id


def sample_task_ids_target_aware(eval_pos: int, batch_size: int, seq_len: int, num_tasks: int, device: str, same_tasks_across_batch=False):
    # split data into target and source tasks with Dirichlet / Categorical sampling
    alphas = torch.ones(num_tasks).to(device) * 2.0
    alphas[0] = 1.0 # target task is less likely to be sampled
    
    # B x N
    if same_tasks_across_batch:
        probabilities = torch.distributions.Dirichlet(alphas).sample()
        task_id = torch.distributions.Categorical(probabilities).sample((seq_len,)).T.to(device)
        task_id = task_id.unsqueeze(0).repeat(batch_size, 1)
    else:
        probabilities = torch.distributions.Dirichlet(alphas).sample((batch_size,))
        task_id = (
            torch.distributions.Categorical(probabilities).sample((seq_len,)).T.to(device)
        )
    # ensure every task is observed during training
    task_id[:, :num_tasks] = torch.arange(num_tasks).repeat(batch_size, 1).to(device)
    # only test on target task
    task_id[:, eval_pos:] = 0

    return task_id


def sample_task_ids_even_eval(eval_pos: int, batch_size: int, seq_len: int, num_tasks: int, device: str, same_tasks_across_batch=False):
    # split data into target and source tasks with Dirichlet / Categorical sampling
    alphas = torch.ones(num_tasks).to(device) * 2.0
    
    # B x N
    if same_tasks_across_batch:
        probabilities = torch.distributions.Dirichlet(alphas).sample()
        task_id = torch.distributions.Categorical(probabilities).sample((seq_len,)).to(device)
        task_id = task_id.unsqueeze(0).repeat(batch_size, 1)
    else:
        probabilities = torch.distributions.Dirichlet(alphas).sample((batch_size,))
        task_id = (
            torch.distributions.Categorical(probabilities).sample((seq_len,)).T.to(device)
        )
    # ensure every task is observed during training
    task_id[:, :num_tasks] = torch.arange(num_tasks).repeat(batch_size, 1).to(device)
    # evenly test on all tasks
    n_eval = seq_len - eval_pos
    even_spread = torch.arange(num_tasks).repeat(batch_size, n_eval // num_tasks + 1)[:, :n_eval]
    task_id[:, eval_pos:] = even_spread

    return task_id


def get_uncorr_task_ids(
    num_tasks: int,
    hyperparameters: dict,
):    
    if "uncorr_prob" in hyperparameters:
        uncorr_prob = hyperparameters["uncorr_prob"]
        uncorr_tasks = (
            torch.where(torch.rand(num_tasks - 1) < uncorr_prob)[0] + 1
        ).tolist()
    elif "num_uncorr_tasks" in hyperparameters:
        num_uncorr_tasks = hyperparameters["num_uncorr_tasks"]
        uncorr_tasks = torch.arange(1, num_tasks)[:num_uncorr_tasks].tolist()
    else:
        uncorr_tasks = []
    
    return uncorr_tasks



def resample_uncorr_tasks(task_id, xs, ys, num_tasks, hyperparameters):
    if "uncorr_prob" in hyperparameters:
        uncorr_prob = hyperparameters["uncorr_prob"]
        uncorr_tasks = (
            torch.where(torch.rand(num_tasks - 1) < uncorr_prob)[0] + 1
        ).tolist()
    elif "num_uncorr_tasks" in hyperparameters:
        num_uncorr_tasks = hyperparameters["num_uncorr_tasks"]
        uncorr_tasks = torch.arange(1, num_tasks)[:num_uncorr_tasks].tolist()
    else:
        uncorr_tasks = []

    # randomly sample uncorrelated tasks
    kernel = RBFKernel().to(xs.device)
    for task in uncorr_tasks:
        assert task > 0
        task_mask = task_id == task
        task_x = xs[task_mask]
        kernel.lengthscale = torch.distributions.Gamma(12, 6).sample().to(xs.device)
        ys[task_mask] = (
            MultivariateNormal(
                torch.zeros(task_x.shape[:-1], device=xs.device), kernel(task_x)
            ).sample()
        ).unsqueeze(-1)

    return ys


def draw_mtgp_samples_original(xs, task_id, input_covar_module, task_covar_matrix):
    task_covar_module = IndexKernelFixed(task_covar_matrix)

    mean_x = ConstantMean()(xs).to(xs.device)
    covar_x = input_covar_module(xs)
    covar_t = task_covar_module(task_id.unsqueeze(-1))
    covar = covar_x.mul(covar_t)

    return MultivariateNormal(mean_x, covar).sample().unsqueeze(-1)


def draw_mtgp_samples(xs, task_id, task_covar_matrix, uncorr_tasks):
    CORR_LENGTHSCALE_A, CORR_LENGTHSCALE_B = 3.0, 6.0
    # UNCORR_LENGTHSCALE_A, UNCORR_LENGTHSCALE_B = 12.0, 6.0
    UNCORR_LENGTHSCALE_A, UNCORR_LENGTHSCALE_B = 3.0, 6.0
    
    B, N, D = xs.shape
    num_tasks = task_covar_matrix.shape[-1]
    n_ind = 1 + len(uncorr_tasks)
    
    if num_tasks == 1:
        mean_x = ConstantMean()(xs).to(xs.device)
        
        lengthscale_prior = gpytorch.priors.GammaPrior(CORR_LENGTHSCALE_A, CORR_LENGTHSCALE_B)
        input_covar_module = RBFKernel().to(xs.device)
        input_covar_module.lengthscale = lengthscale_prior.sample()
        covar_x = input_covar_module(xs)

        return MultivariateNormal(mean_x, covar_x).sample().unsqueeze(-1)
    

    # Add additional "batch" dimension for independent tasks
    xs_expand = xs.unsqueeze(0).expand(n_ind, B, N, D)
    task_id_expand = task_id.unsqueeze(0).unsqueeze(-1).expand(n_ind, B, N, 1)

    mean_x = ConstantMean()(xs_expand).to(xs_expand.device)

    # input kernel: different lengthscale per indepedent task and batch
    a = torch.tensor([CORR_LENGTHSCALE_A] + [UNCORR_LENGTHSCALE_A] * len(uncorr_tasks)).to(xs)
    b = torch.tensor([CORR_LENGTHSCALE_B] + [UNCORR_LENGTHSCALE_B] * len(uncorr_tasks)).to(xs)
    lengthscale_prior = gpytorch.priors.GammaPrior(a, b)
    input_covar_module = RBFKernel(ard_num_dims=D, batch_shape=(n_ind, B)).to(xs.device)
    # final shape: n_ind x B x 1 x D
    input_covar_module.lengthscale = lengthscale_prior.sample((B, D)).unsqueeze(-2).permute(-1, 0, 2, 1)
    covar_x = input_covar_module(xs_expand)

    # task kernel: different correlation per batch
    # todo: different correlation per each batch
    task_covar_matrix = task_covar_matrix.unsqueeze(0).unsqueeze(0).expand(n_ind, B, num_tasks, num_tasks)
    task_covar_module = IndexKernelFixed(task_covar_matrix)
    covar_t = task_covar_module(task_id_expand)

    samples = MultivariateNormal(mean_x, covar_x.mul(covar_t)).sample().unsqueeze(-1)
    
    # combine correlated and uncorrelated samples
    ys = samples[0]
    for i, task in enumerate(uncorr_tasks):
        task_mask = task_id == task
        ys[task_mask] = samples[i+1][task_mask]
    
    return ys


def get_mtgp_for_eval(
    num_features,
    num_tasks,
    lengthscale,
    task_corr,
    num_uncorr_tasks,
    device: str = "cpu",
):
    TARGET_TRAIN = 100
    TARGET_TEST = 500
    SOURCE_TRAIN = 500

    # generate task ids
    task_lengths = [TARGET_TRAIN + TARGET_TEST] + [SOURCE_TRAIN] * (num_tasks - 1)
    task_id = []
    for task, task_length in enumerate(task_lengths):
        task_id.append(torch.ones(task_length) * task)
    task_id = torch.cat(task_id).to(device)

    # generate x values
    xs = torch.rand(len(task_id), num_features).to(device)

    # generate task and input covariances
    input_covar_module = RBFKernel().to(device)
    if lengthscale is None:
        lengthscale = torch.distributions.Gamma(1, 6).sample().to(device)
    input_covar_module.lengthscale = lengthscale

    # task covariance matrix with fixed correlation
    if task_corr is not None:
        task_covar_matrix = torch.full((num_tasks, num_tasks), task_corr, device=device)
        task_covar_matrix.fill_diagonal_(1.0)
    else:
        if num_tasks > 1:
            # sample from LKJ prior
            task_covar_matrix = (
                LKJCovariancePrior(
                    n=num_tasks, eta=1.0, sd_prior=gpytorch.priors.GammaPrior(10.0, 10.0)
                )
                .sample()
                .to(device)
            )
        else:
            task_covar_matrix = torch.eye(num_tasks).to(device)

    ys = draw_mtgp_samples_original(xs, task_id, input_covar_module, task_covar_matrix)
    ys = resample_uncorr_tasks(
        task_id, xs, ys, num_tasks, {"num_uncorr_tasks": num_uncorr_tasks}
    )

    result_xs = []
    result_ys = []
    for task in range(num_tasks):
        task_mask = task_id == task
        result_xs.append(xs[task_mask])
        result_ys.append(ys[task_mask])

    return result_xs[0], result_ys[0], result_xs[1:], result_ys[1:]


# batch_size: number of datasets
# seq_len: number of training samples per dataset
def get_mtgp_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    eval_pos = kwargs["single_eval_pos"]
    seq_len 
    xs = torch.rand(batch_size, seq_len, num_features, device=device)
    task_id = sample_task_ids(batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", True))

    if num_tasks > 1:
        # sample task covariance matrix
        lkj_covar_matrix = (
            LKJCovariancePrior(
                n=num_tasks, eta=1.0, sd_prior=gpytorch.priors.GammaPrior(10.0, 10.0)
            )
            .sample()
            .to(device)
        )
        
        corr_init = hyperparameters.get("corr_init", None)
        if corr_init is not None:
            sampled_corr = torch.distributions.Uniform(corr_init, 1.0).sample().to(device)
            task_covar_matrix = torch.full((num_tasks, num_tasks), sampled_corr).to(lkj_covar_matrix)
            task_covar_matrix.fill_diagonal_(1.0)
            
            blend = torch.rand((1,)).to(device)
            task_covar_matrix = blend * task_covar_matrix + (1 - blend) * lkj_covar_matrix
        else:
            task_covar_matrix = lkj_covar_matrix     
    else:
        task_covar_matrix = torch.eye(num_tasks).to(device)
        
    uncorr_ids = get_uncorr_task_ids(num_tasks, hyperparameters)
    ys = draw_mtgp_samples(xs, task_id, task_covar_matrix, uncorr_ids)
    task_id = task_id.unsqueeze(-1)
    
    noise = torch.distributions.Gamma(1, 5).sample((num_tasks,)).to(device)
    noise_by_task = noise[task_id]
    # noisy_ys = ys + torch.randn_like(ys) * noise_by_task
    noisy_ys = ys.clone()

    return Batch(
        x=xs.transpose(0, 1),
        y=noisy_ys.transpose(0, 1),
        target_y=ys.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    
    
def standardize_per_task(ys, task_id, single_eval_pos, true_ys=None):
    # all task ids should be the same across batches
    assert (task_id.min(dim=0).values == task_id.max(dim=0).values).all(), "task_id is not the same across batches"
    task_id = task_id[0]
        
    standardized_ys = ys.clone()
    if true_ys is not None:
        standardized_true_ys = true_ys.clone()
    
    train_ys = ys[:, :single_eval_pos]
    train_task_id = task_id[:single_eval_pos]
    
    train_mask = torch.zeros_like(ys, dtype=torch.bool)
    train_mask[:, :single_eval_pos] = True
        
    # For each task, compute mean and std based on training data only
    for t in task_id.unique():
        train_task_mask = train_task_id == t
        all_task_mask = task_id == t
        
        # Compute mean and std for training data of this task
        task_train_values = train_ys[:, train_task_mask]
        # batch x seq_len x 1
        task_mean = task_train_values.mean(dim=1, keepdim=True)
        task_std = task_train_values.std(dim=1, keepdim=True)
        
        # Numerical check for 0 or nan
        task_std = torch.where(task_std == 0, torch.ones_like(task_std), task_std)
        task_std = torch.where(task_std.isnan(), torch.ones_like(task_std), task_std)
            
        # Apply standardization to all data points of this task (both train and test)
        standardized_ys[:, all_task_mask] = (ys[:, all_task_mask] - task_mean) / task_std
        if true_ys is not None:
            standardized_true_ys[:, all_task_mask] = (true_ys[:, all_task_mask] - task_mean) / task_std

    if true_ys is not None:
        return standardized_ys, standardized_true_ys
    else:
        return standardized_ys
    
    
def new_standardize_per_task(
    *,
    noisy_ys,
    ys,
    task_id,
    single_eval_pos,
    global_norm="z",
    task_norm=None,
):
    # all task ids should be the same across batches
    assert (
        task_id.min(dim=0).values == task_id.max(dim=0).values
    ).all(), "task_id is not the same across batches, you can set same_tasks_across_batch=True"
    task_id = task_id[0, :]  # shape: seq_len

    def normalize(
        ys_to_fit_on: torch.Tensor, ys_to_transform: torch.Tensor, norm_type: str
    ):
        # both are a tensor of shape batch_size x ?, we want to parallelize across batch_size
        mean = ys_to_fit_on.mean(1, keepdim=True)  # (batch_size, )
        std = ys_to_fit_on.std(1, keepdim=True)  # (batch_size, )
        std = std.where(std.isfinite(), 1.0)
        if norm_type == "z":
            return (ys_to_transform - mean) / std
        if norm_type == "safe_z_0.1":
            safe_std = std.clamp_min(0.1)
            return (ys_to_transform - mean) / safe_std
        elif norm_type == "mean":
            return ys_to_transform - mean
        else:
            raise ValueError(f"Unknown norm type {norm_type}")

    if task_norm is not None:
        # print("num tasks", len(np.unique(task_id.cpu().numpy())))
        # print(
        #     "min train task size",
        #     np.min(
        #         np.unique(task_id[:single_eval_pos].cpu().numpy(), return_counts=True)[
        #             1
        #         ]
        #     ),
        # )
        for tid in task_id.unique():
            local_noisy_ys_for_fitting = noisy_ys[:, :single_eval_pos][
                :, (task_id == tid)[:single_eval_pos]
            ].clone()
            local_noisy_ys = noisy_ys[:, (task_id == tid)]
            local_ys = ys[:, (task_id == tid)]

            ys[:, (task_id == tid)] = normalize(
                local_noisy_ys_for_fitting, local_ys, task_norm
            )
            noisy_ys[:, (task_id == tid)] = normalize(
                local_noisy_ys_for_fitting, local_noisy_ys, task_norm
            )

    if global_norm is not None:
        # print("doing global norm with", global_norm)
        noisy_ys_for_fitting = noisy_ys[:, :single_eval_pos].clone()

        ys[:] = normalize(noisy_ys_for_fitting, ys, global_norm)
        noisy_ys[:] = normalize(noisy_ys_for_fitting, noisy_ys, global_norm)

    return noisy_ys, ys

    
def target_aware_mtgp_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    single_eval_pos = kwargs.get("single_eval_pos", seq_len // 2)
    xs = torch.rand(batch_size, seq_len, num_features, device=device)
    task_id = sample_task_ids_target_aware(single_eval_pos, batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", True))

    if num_tasks > 1:
        # sample task covariance matrix
        lkj_covar_matrix = (
            LKJCovariancePrior(
                n=num_tasks, eta=1.0, sd_prior=gpytorch.priors.GammaPrior(10.0, 10.0)
            )
            .sample()
            .to(device)
        )
        
        corr_init = hyperparameters.get("corr_init", None)
        if corr_init is not None:
            # bias towards correlations between corr_init and 1.0
            sampled_corr = torch.distributions.Uniform(corr_init, 1.0).sample().to(device)
            task_covar_matrix = torch.full((num_tasks, num_tasks), sampled_corr).to(lkj_covar_matrix)
            task_covar_matrix.fill_diagonal_(1.0)
            
            blend = torch.rand((1,)).to(device)
            task_covar_matrix = blend * task_covar_matrix + (1 - blend) * lkj_covar_matrix
        else:
            task_covar_matrix = lkj_covar_matrix     
    else:
        task_covar_matrix = torch.eye(num_tasks).to(device)
        
    # sample number of uncorrelated tasks uniformly from 0 to T-1
    if num_tasks > 1:
        n_uncorr = torch.randint(num_tasks - 1, (1,)).item()
        uncorr_ids = torch.randperm(num_tasks - 1)[:n_uncorr]
        uncorr_ids = (uncorr_ids + 1).tolist() # task 0 cannot be uncorr
    else:
        uncorr_ids = []
    
    ys = draw_mtgp_samples(xs, task_id, task_covar_matrix, uncorr_ids)
    
    sigma = torch.distributions.LogNormal(-4.0, 1.0).sample(ys.shape).to(device)
    noisy_ys = ys.clone() + torch.randn_like(ys) * sigma
    
    noisy_ys, ys = new_standardize_per_task(
        noisy_ys=noisy_ys,
        ys=ys,
        task_id=task_id,
        single_eval_pos=single_eval_pos,
        global_norm=hyperparameters.get("global_y_norm", None),
        task_norm=hyperparameters.get("task_y_norm", "safe_z_0.1"),
    )
    
    task_id = task_id.unsqueeze(-1)

    return Batch(
        x=xs.transpose(0, 1),
        y=noisy_ys.transpose(0, 1),
        target_y=ys.transpose(0, 1),
        task_id=task_id.transpose(0, 1),
    )
    
    
def gen_mtgp_equal_eval_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    single_eval_pos = kwargs.get("single_eval_pos", seq_len // 2)
    xs = torch.rand(batch_size, seq_len, num_features, device=device)
    task_id = sample_task_ids_even_eval(single_eval_pos, batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", True))

    if num_tasks > 1:
        # sample task covariance matrix
        lkj_covar_matrix = (
            LKJCovariancePrior(
                n=num_tasks, eta=1.0, sd_prior=gpytorch.priors.GammaPrior(10.0, 10.0)
            )
            .sample()
            .to(device)
        )
        
        corr_init = hyperparameters.get("corr_init", None)
        if corr_init is not None:
            # bias towards correlations between corr_init and 1.0
            sampled_corr = torch.distributions.Uniform(corr_init, 1.0).sample().to(device)
            task_covar_matrix = torch.full((num_tasks, num_tasks), sampled_corr).to(lkj_covar_matrix)
            task_covar_matrix.fill_diagonal_(1.0)
            
            blend = torch.rand((1,)).to(device)
            task_covar_matrix = blend * task_covar_matrix + (1 - blend) * lkj_covar_matrix
        else:
            task_covar_matrix = lkj_covar_matrix     
    else:
        task_covar_matrix = torch.eye(num_tasks).to(device)
        
    # sample number of uncorrelated tasks uniformly from 0 to T-1
    if num_tasks > 1:
        n_uncorr = torch.randint(num_tasks - 1, (1,)).item()
        uncorr_ids = torch.randperm(num_tasks - 1)[:n_uncorr]
        uncorr_ids = (uncorr_ids + 1).tolist() # task 0 cannot be uncorr
    else:
        uncorr_ids = []
    
    ys = draw_mtgp_samples(xs, task_id, task_covar_matrix, uncorr_ids)
    
    sigma = torch.distributions.LogNormal(-4.0, 1.0).sample(ys.shape).to(device)
    noisy_ys = ys.clone() + torch.randn_like(ys) * sigma
    
    noisy_ys, ys = new_standardize_per_task(
        noisy_ys=noisy_ys,
        ys=ys,
        task_id=task_id,
        single_eval_pos=single_eval_pos,
        global_norm=hyperparameters.get("global_y_norm", None),
        task_norm=hyperparameters.get("task_y_norm", "safe_z_0.1"),
    )

    task_id = task_id.unsqueeze(-1)
    
    return Batch(
        x=xs.transpose(0, 1),
        y=noisy_ys.transpose(0, 1),
        target_y=ys.transpose(0, 1),
        task_id=task_id.transpose(0, 1),
    )
    
    
def gen_mtgp_low_rank(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters: dict,
    device: str = default_device,
    **kwargs,
):
    single_eval_pos = kwargs.get("single_eval_pos", seq_len // 2)
    xs = torch.rand(batch_size, seq_len, num_features, device=device)
    if hyperparameters.get("target_only_loss", False):
        task_id = sample_task_ids_target_aware(single_eval_pos, batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", True))
    else:
        task_id = sample_task_ids_even_eval(single_eval_pos, batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", True))

    rank = torch.randint(2, hyperparameters.get("max_rank", 5)+1, (1,)).item()
    alpha = hyperparameters.get("decay_alpha", 0.2)
    if num_tasks > 1:
        weights = torch.exp(-alpha * torch.arange(1, num_tasks).float())
        n_ind = torch.multinomial(weights, 1).item()
    else:
        n_ind = 0
    B = batch_size
    D = num_features

    # Sample mixing weights from Dirichlet
    concentration = 1.0
    concentration_vec = torch.full((rank,), concentration, device=device)

    W_full = torch.zeros((num_tasks, rank + n_ind), device=device)
    # split weights between rank
    W_rank = torch.distributions.Dirichlet(concentration_vec).sample((num_tasks,)).to(device)
    W_full[:, :rank] = W_rank
    # randomly non-target tasks to be independent
    independent_tasks = torch.randperm(num_tasks - 1)[:n_ind] + 1
    for i, ind in enumerate(independent_tasks):
        W_full[ind, :] = 0.0
        W_full[ind, -(i+1)] = 1.0

    # Create expanded inputs: batch, (rank), seq, features)
    xs_expanded = xs.unsqueeze(1).expand((B, rank + n_ind, seq_len, D))

    # Create task IDs for all latent functions
    latent_task_ids = torch.arange(rank, device=device).unsqueeze(0).unsqueeze(-1)
    latent_task_ids = latent_task_ids.expand(batch_size, rank, seq_len)
    latent_task_ids_mega = latent_task_ids.reshape(batch_size * rank, seq_len)

    CORR_LENGTHSCALE_A, CORR_LENGTHSCALE_B = 3.0, 6.0
    UNCORR_LENGTHSCALE_A, UNCORR_LENGTHSCALE_B = 3.0, 6.0
    a = torch.tensor([CORR_LENGTHSCALE_A] + [UNCORR_LENGTHSCALE_A] * n_ind).to(xs)
    b = torch.tensor([CORR_LENGTHSCALE_B] + [UNCORR_LENGTHSCALE_B] * n_ind).to(xs)

    # Create lengthscale prior
    lengthscale_prior = gpytorch.priors.GammaPrior(a, b)
    # original B x D x (1 + n_ind) => B x (1 + n_ind) x D
    lengthscale_sample = lengthscale_prior.sample((B, D)).transpose(1, 2)

    kernel = RBFKernel(ard_num_dims=D, batch_shape=torch.Size((B, rank + n_ind))).to(xs.device)
    # lengthscale should have shape B x (rank + n_ind) x 1 x D
    lengthscale = torch.ones((B, rank+n_ind, 1, D), device=xs.device)
    # make sure rank lengthscales are the same
    rank_lengthscales = lengthscale_sample[:, 0].unsqueeze(1).unsqueeze(1).expand(B, rank, 1, D).clone()
    lengthscale[:, :rank] = rank_lengthscales
    # use remaining lengthscales for independent draws
    lengthscale[:, rank:] = lengthscale_sample[:, 1:].unsqueeze(-2)

    kernel.lengthscale = lengthscale
    K = kernel(xs_expanded)
    mean_x = ConstantMean().to(xs.device)(xs_expanded)
    # size B x (rank + n_ind) x seq_len
    latent_functions = MultivariateNormal(mean_x, K).sample()

    # Apply mixing (same across all batches since task IDs are same)
    task_id_seq = task_id[0]  # Pattern from first batch
    mixing_weights_seq = W_full[task_id_seq].T  # rank + n_ind x num_tasks

    # Mix latent functions
    ys = torch.sum(latent_functions * mixing_weights_seq.unsqueeze(0), dim=1).unsqueeze(-1)
    
    sigma = torch.distributions.LogNormal(-4.0, 1.0).sample(ys.shape).to(device)
    noisy_ys = ys.clone() + torch.randn_like(ys) * sigma
    
    noisy_ys, ys = new_standardize_per_task(
        noisy_ys=noisy_ys,
        ys=ys,
        task_id=task_id,
        single_eval_pos=single_eval_pos,
        global_norm=hyperparameters.get("global_y_norm", None),
        task_norm=hyperparameters.get("task_y_norm", "safe_z_0.1"),
    )

    task_id = task_id.unsqueeze(-1)
    
    return Batch(
        x=xs.transpose(0, 1),
        y=noisy_ys.transpose(0, 1),
        target_y=ys.transpose(0, 1),
        task_id=task_id.transpose(0, 1),
    )



def get_hpo_batch_fn(
    train=True,
    max_features=6,
    min_num_tasks=6,
    seq_len=200,
    device=default_device,
    **kwargs,
):
    DATA_PATHS = {
        "test": "/home/yl9959/mtpfn/datasets/hpob-data/meta-test-dataset.json",
        "train-augmented": "/home/yl9959/mtpfn/datasets/hpob-data/meta-train-dataset-augmented.json",
    }

    if train:
        data = json.load(open(DATA_PATHS["train-augmented"], "r"))
    else:
        data = json.load(open(DATA_PATHS["test"], "r"))

    domain_weight = []
    domain_data = []
    for domain in data:
        hpo_runs_weight = []
        hpo_runs_x = []
        hpo_runs_y = []
        for hpo_run in data[domain]:
            x = torch.tensor(data[domain][hpo_run]["X"], device=device)
            # skip this domain if it has more than max_features features
            if x.shape[-1] > max_features:
                break

            if x.shape[0] < seq_len:
                continue

            hpo_runs_weight.append(len(x))
            hpo_runs_x.append(x.to(device))
            hpo_runs_y.append(torch.tensor(data[domain][hpo_run]["y"], device=device))

        # skip this domain if it doesn't have enough tasks
        if len(hpo_runs_weight) <= min_num_tasks:
            continue
        domain_weight.append(sum(hpo_runs_weight))
        domain_data.append(
            (torch.tensor(hpo_runs_weight, device=device), hpo_runs_x, hpo_runs_y)
        )
        
    domain_weight = torch.tensor(domain_weight)
    domain_weight = domain_weight / domain_weight.sum()
    # add uniform probability
    domain_weight = domain_weight * 0.5 + 1 / len(domain_weight) * 0.5

    def get_batch(
        batch_size: int,
        seq_len: int,
        num_features: int,
        max_num_tasks: int,
        num_tasks: int,
        hyperparameters=None,
        device: str = default_device,
        **kwargs,
    ):
        domain_idx = torch.multinomial(domain_weight, 1)
        task_weight, task_x, task_y = domain_data[domain_idx]
        task_data_indices = torch.multinomial(task_weight.float(), num_tasks)

        task_id = sample_task_ids(batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", False))
        # task_id = sample_task_ids(batch_size, seq_len, num_tasks, device)

        xs = torch.zeros(batch_size, seq_len, task_x[0].shape[-1], device=device)
        ys = torch.zeros(batch_size, seq_len, 1, device=device)
        for i in range(num_tasks):
            relevant_indices = task_id == i
            possible_xs, possible_ys = (
                task_x[task_data_indices[i]],
                task_y[task_data_indices[i]],
            )
            # randomly select per batch
            for b in range(batch_size):
                relevant_batch_indices = relevant_indices[b]
                indices = torch.randperm(len(possible_xs))[
                    : relevant_batch_indices.sum()
                ]
                xs[b, relevant_batch_indices] = possible_xs[indices].to(xs.device)
                ys[b, relevant_batch_indices] = possible_ys[indices].to(ys.device)

        return Batch(
            x=xs.transpose(0, 1).to(device),
            y=ys.transpose(0, 1).to(device),
            target_y=ys.transpose(0, 1).clone().to(device),
            task_id=task_id.unsqueeze(-1).transpose(0, 1).to(device),
        )

    return get_batch


def get_pd1_surrogate_batch_fn(
    train=True,
    max_features=6,
    min_num_tasks=6,
    seq_len=200,
    device=default_device,
    **kwargs,
):
    
    data = pickle.load(open("/home/yl9959/mtpfn/datasets/pd1.pickle", "rb"))

    if not os.path.exists("/home/yl9959/mtpfn/pd1_gp_surrogates.pth"):
        
        # group by data
        data_grouped = data.groupby("study_group")
        x_cols = [
            'hps.lr_hparams.decay_steps_factor',
            'hps.lr_hparams.initial_value',
            'hps.lr_hparams.power',
            'hps.opt_hparams.momentum', 
        ]
        y_col = 'best_valid/error_rate'

        x_bounds = torch.tensor([
            data[x_cols].min().values,
            data[x_cols].max().values
        ])

        # for each group
        gps = []
        for group_name, group_data in data_grouped:
            print("Fitting GP for group", group_name)
            train_x = torch.tensor(group_data[x_cols].values)
            train_x = (train_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
            train_y = torch.tensor(group_data[y_col].values).unsqueeze(-1)
            
            train_yvar = torch.ones_like(train_y) * 1e-5
            gp = SingleTaskGP(train_x, train_y, train_yvar, outcome_transform=None)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                fit_gpytorch_mll_torch(mll)
            gps.append((group_name, gp))
            
        state_dicts = {name: gp.state_dict() for name, gp in gps}
        torch.save(state_dicts, "/home/yl9959/mtpfn/pd1_gp_surrogates.pth")

    gp_state_dicts = torch.load("/home/yl9959/mtpfn/pd1_gp_surrogates.pth")
    gps = []
    for data_name, state_dict in gp_state_dicts.items():
        if "cifar" in data_name:
            continue
        gp = SingleTaskGP(torch.zeros(1, 4), torch.zeros(1, 1), torch.ones(1, 1) * 1e-5, outcome_transform=None)
        gp.load_state_dict(state_dict)
        gps.append(gp)


    def get_batch(
        batch_size: int,
        seq_len: int,
        num_features: int,
        max_num_tasks: int,
        num_tasks: int,
        hyperparameters=None,
        device: str = default_device,
        **kwargs,
    ):
        task_id = sample_task_ids(batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", False))

        xs = torch.rand(batch_size, seq_len, 4, device=device)
        ys = torch.zeros(batch_size, seq_len, 1, device=device)
        
        surrogate_gps = torch.randperm(len(gps))[:num_tasks]
        for i in range(num_tasks):
            task_mask = task_id == i
            task_xs = xs[task_mask]
            gp = gps[surrogate_gps[i]].to(device)
            with torch.no_grad():
                ys[task_mask] = gp.posterior(task_xs).sample()
            
        return Batch(
            x=xs.transpose(0, 1).to(device),
            y=ys.transpose(0, 1).to(device),
            target_y=ys.transpose(0, 1).clone().to(device),
            task_id=task_id.unsqueeze(-1).transpose(0, 1).to(device),
        )

    return get_batch


def get_pd1_eval_batch_fn(
    train=True,
    max_features=6,
    min_num_tasks=6,
    seq_len=200,
    device=default_device,
    **kwargs,
):

    data = pickle.load(open("/home/yl9959/mtpfn/datasets/pd1.pickle", "rb"))

    # group by data
    data_grouped = data.groupby("study_group")
    x_cols = [
        'hps.lr_hparams.decay_steps_factor',
        'hps.lr_hparams.initial_value',
        'hps.lr_hparams.power',
        'hps.opt_hparams.momentum', 
    ]
    y_col = 'best_valid/error_rate'
    
    x_bounds = torch.tensor([
        data[x_cols].min().values,
        data[x_cols].max().values
    ])
    
    all_xs = []
    all_ys = []
    for group_name, group_data in data_grouped:
        if "cifar" in group_name: # eval on cifar
            train_x = torch.tensor(group_data[x_cols].values)
            train_x = (train_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
            train_y = torch.tensor(group_data[y_col].values).unsqueeze(-1)
            all_xs.append(train_x)
            all_ys.append(train_y)
    
    def get_batch(
        batch_size: int,
        seq_len: int,
        num_features: int,
        max_num_tasks: int,
        num_tasks: int,
        hyperparameters=None,
        device: str = default_device,
        **kwargs,
    ):
        assert num_tasks <= 4
        task_id = sample_task_ids(batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", False))
        
        # dummy values to be replaced
        xs = torch.zeros(batch_size, seq_len, num_features, device=device)
        ys = torch.zeros(batch_size, seq_len, 1, device=device)

        for b in range(batch_size):
            for task in range(num_tasks):
                task_mask = task_id[b] == task
                task_x = all_xs[task]
                task_y = all_ys[task]
                indices = torch.randperm(len(task_x))[:seq_len]
                size = xs[b, task_mask].shape[0]
                xs[b, task_mask] = task_x[indices][:size].to(xs)
                ys[b, task_mask] = task_y[indices][:size].to(ys)
                
        return Batch(
            x=xs.transpose(0, 1).to(device),
            y=ys.transpose(0, 1).to(device),
            target_y=ys.transpose(0, 1).clone().to(device),
            task_id=task_id.unsqueeze(-1).transpose(0, 1).to(device),
        )

    return get_batch

    
def combine_batch(functions, weight):
    def get_batch(
        *args,
        **kwargs,
    ):
        random_index = torch.multinomial(weight, 1)[0]
        return functions[random_index](*args, **kwargs)

    return get_batch
    

# batch_size: number of datasets
# seq_len: number of training samples per dataset
def get_trios_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    xs = torch.rand(batch_size, seq_len, num_features, device=device)
    ys = xs.sum(-1, keepdim=True)
    
    assert hyperparameters.get("same_tasks_across_batch", False)
    task_id = sample_task_ids(batch_size, seq_len, num_tasks, device, hyperparameters.get("same_tasks_across_batch", False))[0]
    
    task_order = torch.randperm(num_tasks)
    # assert same tasks across batch

    num_trios = num_tasks // 3
    
    for trio in range(num_trios):
        task_one = task_order[trio * 3]
        task_two = task_order[trio * 3 + 1]
        task_three = task_order[trio * 3 + 2]
        task_mask = (task_id == task_one) | (task_id == task_two) | (task_id == task_three)
        
        task_covar_matrix = torch.full((num_tasks, num_tasks), 0.9).to(xs)
        task_covar_matrix.fill_diagonal_(1.0)
            
        y_trio = draw_mtgp_samples(xs[:, task_mask], task_id[task_mask], task_covar_matrix, uncorr_tasks=[])
        ys[:, task_mask] = y_trio
        
    remaining = num_tasks - num_trios * 3
    if remaining > 0:
        task_mask = torch.zeros_like(task_id, dtype=torch.bool)
        for task_id in task_order[num_trios * 3:]:
            task_mask |= task_id == task_id
        task_covar_matrix = torch.full((remaining, remaining), 0.1).to(xs)
        task_covar_matrix.fill_diagonal_(1.0)
        ys[:, task_mask] = draw_mtgp_samples(xs[:, task_mask], task_id[task_mask], task_covar_matrix, uncorr_tasks=[])
        
    # add back batch dimension
    task_id = task_id.unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
    
    noise = torch.distributions.Gamma(1, 5).sample((num_tasks,)).to(device)
    noise_by_task = noise[task_id]
    # noisy_ys = ys + torch.randn_like(ys) * noise_by_task
    noisy_ys = ys.clone()

    return Batch(
        x=xs.transpose(0, 1),
        y=noisy_ys.transpose(0, 1),
        target_y=ys.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    
    
def get_hpobench_batch_fn(
    hpobench_task="lr",
    train=True,
    device=default_device,
    **kwargs,
):
    
    data = pickle.load(open(f"/home/lily_l/private_multitask_pfn/datasets/hpobench_{hpobench_task}.pkl", "rb"))
    if hpobench_task == "lr":
        eval_keys = [146822, 146818, 168908, 53]
        dim = 2
    elif hpobench_task == "svm":
        eval_keys = [146822, 146818, 168908, 53]
        dim = 2
    else:
        raise ValueError("Unknown hpobench task")
        
    if train:
        batch_keys = data.keys() - eval_keys # train keys
    else:
        batch_keys = eval_keys

    results_x, results_y = get_torch_format_hpobench(data[eval_keys[0]], [data[key] for key in batch_keys], dim, hpobench_task)[2:]
    results_x = results_x
    results_y = results_y

    def get_batch(
        batch_size: int,
        seq_len: int,
        num_features: int,
        max_num_tasks: int,
        num_tasks: int,
        hyperparameters=None,
        device: str = default_device,
        **kwargs,
    ):
        assert num_tasks == 1
        task_id = torch.zeros(batch_size, seq_len, device=device).long()
        
        train_task = torch.randint(len(results_x), (1,))
        all_xs, all_ys = results_x[train_task], results_y[train_task]
        
        indices = torch.stack([torch.randperm(len(all_xs))[:seq_len] for _ in range(batch_size)])
        xs = all_xs[indices]
        ys = all_ys[indices]
            
        return Batch(
            x=xs.transpose(0, 1).to(device),
            y=ys.transpose(0, 1).to(device),
            target_y=ys.transpose(0, 1).clone().to(device),
            task_id=task_id.unsqueeze(-1).transpose(0, 1).to(device),
        )

    return get_batch

