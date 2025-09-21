from __future__ import annotations

import torch

from multitask.configs import (
    MultitaskTrainingPlan,
    build_multitask_main_config,
    estimate_target_borders,
)


def _small_plan(num_tasks: int = 3) -> MultitaskTrainingPlan:
    return MultitaskTrainingPlan(
        num_tasks=num_tasks,
        batch_size=2,
        seq_len=32,
        single_eval_pos=20,
        num_features=4,
        steps_per_epoch=2,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        emsize=32,
        nhid=64,
        nlayers=2,
        nhead=4,
        num_histogram_buckets=8,
        sample_batches_for_borders=4,
        train_mixed_precision=False,
        device="cpu",
    )


def test_build_multitask_main_config_generates_valid_config():
    torch.manual_seed(0)
    plan = _small_plan(num_tasks=3)
    config = build_multitask_main_config(plan)

    assert config.model.use_hierarchical_attention is True
    assert config.batch_shape_sampler.min_num_tasks == plan.num_tasks
    assert config.batch_shape_sampler.max_num_tasks == plan.num_tasks

    get_batch = config.priors[0].create_get_batch_method()
    batch = get_batch(
        batch_size=plan.batch_size,
        seq_len=plan.seq_len,
        num_features=plan.num_features,
        single_eval_pos=plan.single_eval_pos,
    )

    assert batch.task_indices is not None
    assert batch.task_indices.min().item() >= -1
    assert batch.task_indices.max().item() < plan.num_tasks
    assert batch.num_tasks == plan.num_tasks


def test_estimate_target_borders_is_sorted_and_finite():
    torch.manual_seed(0)
    plan = _small_plan(num_tasks=4)
    borders = estimate_target_borders(plan)

    assert len(borders) >= 2
    assert all(torch.isfinite(torch.tensor(borders)))
    assert borders == sorted(borders)
