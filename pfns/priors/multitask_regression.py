from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from pfns.priors.prior import Batch
from pfns.utils import default_device


@dataclass
class MultiTaskPriorConfig:
    """Lightweight configuration container for the multitask regression prior."""

    num_tasks: int
    weight_std: float = 1.0
    task_offset_std: float = 0.5
    bias_std: float = 0.1
    observation_noise: float = 0.05
    device: str = default_device

    def create_get_batch_method(self):
        def _get_batch(**kwargs):
            return get_batch(
                **kwargs,
                num_tasks=self.num_tasks,
                weight_std=self.weight_std,
                task_offset_std=self.task_offset_std,
                bias_std=self.bias_std,
                observation_noise=self.observation_noise,
                device=self.device,
            )

        return _get_batch


def _allocate_counts(total: int, num_tasks: int, device: torch.device) -> torch.Tensor:
    if total <= 0:
        return torch.zeros(num_tasks, dtype=torch.long, device=device)

    if total >= num_tasks:
        base = total // num_tasks
        counts = torch.full((num_tasks,), base, dtype=torch.long, device=device)
        remainder = total % num_tasks
        if remainder:
            counts[:remainder] += 1
        return counts

    counts = torch.zeros(num_tasks, dtype=torch.long, device=device)
    counts[:total] = 1
    return counts


@torch.no_grad()
def get_batch(
    *,
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: Optional[int],
    num_tasks: Optional[int] = None,
    device: str = default_device,
    weight_std: float = 1.0,
    task_offset_std: float = 0.5,
    bias_std: float = 0.1,
    observation_noise: float = 0.05,
    **_: object,
) -> Batch:
    """Sample a batch of multitask regression problems."""

    if single_eval_pos is None:
        raise ValueError("single_eval_pos must be provided for multitask batches.")

    if num_tasks is None:
        num_tasks = 1

    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")

    torch_device = torch.device(device)

    x = torch.randn(batch_size, seq_len, num_features, device=torch_device)
    y = torch.zeros(batch_size, seq_len, 1, device=torch_device)
    target_y = torch.zeros_like(y)
    task_index_tensor = torch.full(
        (batch_size, seq_len), -1, dtype=torch.long, device=torch_device
    )

    test_len = seq_len - single_eval_pos

    for batch_id in range(batch_size):
        shared_weight = torch.randn(num_features, 1, device=torch_device) * weight_std
        task_weights = shared_weight.unsqueeze(0).repeat(num_tasks, 1, 1)
        task_weights += torch.randn_like(task_weights) * task_offset_std
        task_bias = torch.randn(num_tasks, 1, device=torch_device) * bias_std

        train_counts = _allocate_counts(single_eval_pos, num_tasks, torch_device)
        test_counts = _allocate_counts(test_len, num_tasks, torch_device)

        task_order = torch.randperm(num_tasks, device=torch_device)

        train_ids = []
        for task_id in task_order:
            count = int(train_counts[task_id].item())
            if count == 0:
                continue
            train_ids.append(torch.full((count,), task_id.item(), device=torch_device))
        if train_ids:
            train_ids = torch.cat(train_ids)
            train_ids = train_ids[torch.randperm(train_ids.numel(), device=torch_device)]
        else:
            train_ids = torch.empty(0, dtype=torch.long, device=torch_device)

        test_ids = []
        for task_id in task_order:
            count = int(test_counts[task_id].item())
            if count == 0:
                continue
            test_ids.append(torch.full((count,), task_id.item(), device=torch_device))
        if test_ids:
            test_ids = torch.cat(test_ids)
            test_ids = test_ids[torch.randperm(test_ids.numel(), device=torch_device)]
        else:
            test_ids = torch.empty(0, dtype=torch.long, device=torch_device)

        all_ids = torch.cat((train_ids, test_ids))
        if all_ids.numel() != seq_len:
            padding = seq_len - all_ids.numel()
            if padding > 0:
                pad_ids = torch.full((padding,), -1, device=torch_device)
                all_ids = torch.cat((all_ids, pad_ids))
            else:
                all_ids = all_ids[:seq_len]

        task_index_tensor[batch_id] = all_ids

        for task_id in range(num_tasks):
            mask = all_ids == task_id
            if not bool(mask.any()):
                continue
            outputs = x[batch_id, mask] @ task_weights[task_id]
            outputs = outputs + task_bias[task_id]
            y[batch_id, mask, 0] = outputs.squeeze(-1)

        noise = torch.randn_like(y[batch_id]) * observation_noise
        y[batch_id] += noise
        target_y[batch_id] = y[batch_id]

    return Batch(
        x=x,
        y=y,
        target_y=target_y,
        single_eval_pos=single_eval_pos,
        task_indices=task_index_tensor,
        num_tasks=num_tasks,
    )
