from __future__ import annotations

import statistics
import time
from dataclasses import replace
from typing import Iterable

import torch

from pfns.priors import multitask_regression
from pfns.train import compute_losses

from .configs import MultitaskTrainingPlan, build_multitask_main_config


def _run_single_pass(
    model,
    batch,
    *,
    single_eval_pos: int,
    num_tasks: int | None,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(
            x=batch.x,
            y=batch.y[:, :single_eval_pos],
            task_indices=batch.task_indices,
            num_tasks=num_tasks,
        )
    return output


def run_runtime_benchmark(
    task_counts: Iterable[int],
    *,
    plan: MultitaskTrainingPlan | None = None,
    warmup: int = 1,
    runs: int = 3,
    device: str | None = None,
) -> dict[int, dict[str, float]]:
    """Benchmark latency and loss across different task counts."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    base_plan = plan or MultitaskTrainingPlan()
    results: dict[int, dict[str, float]] = {}

    for num_tasks in task_counts:
        current_plan = base_plan.spawn_for_tasks(num_tasks)
        if device is not None:
            current_plan = replace(current_plan, device=device)

        config = build_multitask_main_config(current_plan)
        model = config.model.create_model().to(current_plan.device)
        model.eval()

        batch_shape = config.batch_shape_sampler.sample_batch_shape(epoch=0, step=0)
        sampler_kwargs = batch_shape.as_get_batch_kwargs()
        prior_kwargs = current_plan.prior_kwargs().copy()
        # Avoid passing num_tasks twice when the sampler already specifies it.
        if sampler_kwargs.get("num_tasks") is not None:
            prior_kwargs.pop("num_tasks", None)

        batch = multitask_regression.get_batch(
            **sampler_kwargs,
            **prior_kwargs,
        )
        batch.x = batch.x.to(current_plan.device)
        batch.y = batch.y.to(current_plan.device)
        batch.target_y = batch.target_y.to(current_plan.device)
        if batch.task_indices is not None:
            batch.task_indices = batch.task_indices.to(current_plan.device)

        timings: list[float] = []
        loss_values: list[float] = []

        total_runs = warmup + runs
        for idx in range(total_runs):
            start = time.perf_counter()
            output = _run_single_pass(
                model,
                batch,
                single_eval_pos=batch_shape.single_eval_pos,
                num_tasks=batch.num_tasks,
            )
            elapsed = time.perf_counter() - start
            if idx >= warmup:
                timings.append(elapsed)

                targets = batch.target_y[:, batch_shape.single_eval_pos :]
                losses = compute_losses(
                    output,
                    targets,
                    model.criterion,
                    n_targets_per_input=1,
                )
                loss_values.append(losses.mean().item())

        results[num_tasks] = {
            "mean_latency": statistics.mean(timings) if timings else 0.0,
            "stdev_latency": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
            "mean_loss": statistics.mean(loss_values) if loss_values else float("nan"),
        }

    return results


__all__ = ["run_runtime_benchmark"]
