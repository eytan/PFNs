from __future__ import annotations

import torch

from multitask.benchmark import run_runtime_benchmark
from multitask.configs import MultitaskTrainingPlan


def _benchmark_plan() -> MultitaskTrainingPlan:
    return MultitaskTrainingPlan(
        num_tasks=3,
        batch_size=2,
        seq_len=28,
        single_eval_pos=18,
        num_features=3,
        steps_per_epoch=2,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        emsize=32,
        nhid=64,
        nlayers=2,
        nhead=4,
        num_histogram_buckets=8,
        sample_batches_for_borders=3,
        device="cpu",
    )


def test_runtime_benchmark_returns_latency_and_loss():
    torch.manual_seed(123)
    plan = _benchmark_plan()
    results = run_runtime_benchmark([3, 5], plan=plan, warmup=0, runs=1, device="cpu")

    assert set(results.keys()) == {3, 5}
    for metrics in results.values():
        assert metrics["mean_latency"] >= 0.0
        assert "mean_loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["mean_loss"]))
