from __future__ import annotations

from typing import Tuple

import torch
from torch.distributions import LKJCholesky, MultivariateNormal

from pfns.utils import default_device

from .prior import Batch


def _sample_task_covariance(
    num_tasks: int,
    *,
    device: torch.device,
    concentration: float,
) -> torch.Tensor:
    cholesky = LKJCholesky(num_tasks, concentration=concentration).sample().to(device)
    corr = cholesky @ cholesky.T
    scales = torch.distributions.Gamma(2.0, 1.0).sample((num_tasks,)).to(device)
    scale_matrix = torch.diag(scales.sqrt())
    return scale_matrix @ corr @ scale_matrix


def _ensure_cholesky(
    matrix: torch.Tensor,
    jitter: float = 1e-5,
    max_tries: int = 6,
) -> torch.Tensor:
    """Return a stable Cholesky factor for the provided covariance matrix.

    Numerical noise can make the sampled kernel non positive-definite, so we
    symmetrize the matrix and add diagonal jitter until decomposition succeeds.
    """

    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    attempt = (matrix + matrix.transpose(0, 1)) / 2
    for _ in range(max_tries):
        chol, info = torch.linalg.cholesky_ex(attempt)
        if int(info.item()) == 0:
            return chol
        attempt = attempt + jitter * eye
        jitter *= 10

    raise RuntimeError("Failed to compute a valid Cholesky factor for task covariance")


@torch.no_grad()
def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    *,
    single_eval_pos: int,
    device: torch.device = default_device,
    min_tasks: int = 2,
    max_tasks: int = 5,
    task_eta: float = 2.0,
    p_unrelated: float = 0.2,
    lengthscale_range: Tuple[float, float] = (0.3, 1.5),
    noise_scale: float = 0.05,
    **_: object,
) -> Batch:
    if min_tasks < 1 or max_tasks < min_tasks:
        raise ValueError("Invalid task range provided")

    x = torch.randn(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    target_y = torch.zeros_like(y)
    task_partition = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        num_tasks = torch.randint(min_tasks, max_tasks + 1, (1,), device=device).item()
        tasks = torch.randint(0, num_tasks, (seq_len,), device=device)
        for task_id in range(num_tasks):
            if (tasks == task_id).sum() == 0:
                tasks[task_id % seq_len] = task_id
        task_partition[batch_idx] = tasks

        lengthscale = torch.empty(1, device=device).uniform_(*lengthscale_range).item()
        diffs = x[batch_idx].unsqueeze(1) - x[batch_idx].unsqueeze(0)
        sqdist = diffs.pow(2).sum(-1)
        kernel_x = torch.exp(-0.5 * sqdist / (lengthscale**2))

        task_cov = _sample_task_covariance(
            num_tasks,
            device=device,
            concentration=task_eta,
        )

        if torch.rand(1, device=device).item() < p_unrelated:
            unrelated = torch.rand(num_tasks, device=device) < 0.5
            if unrelated.all():
                unrelated[torch.randint(0, num_tasks, (1,), device=device).item()] = False
            for idx, flag in enumerate(unrelated.tolist()):
                if flag:
                    task_cov[idx, :] = 0
                    task_cov[:, idx] = 0
                    task_cov[idx, idx] = torch.clamp(task_cov[idx, idx], min=1e-3)

        cov_latent = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            cov_latent[i] = task_cov[tasks[i], tasks] * kernel_x[i]

        cov_latent = cov_latent + 1e-5 * torch.eye(seq_len, device=device)
        chol = _ensure_cholesky(cov_latent)
        mvn = MultivariateNormal(
            torch.zeros(seq_len, device=device),
            scale_tril=chol,
        )
        latent = mvn.sample()
        observed = latent + noise_scale * torch.randn(seq_len, device=device)

        y[batch_idx, :, 0] = observed
        target_y[batch_idx, :, 0] = latent

    return Batch(
        x=x,
        y=y,
        target_y=target_y,
        task_partition=task_partition,
        single_eval_pos=single_eval_pos,
    )

