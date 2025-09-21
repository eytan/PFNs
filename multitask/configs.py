from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import torch

from pfns.batch_shape_sampler import BatchShapeSamplerConfig
from pfns.model.criterions import BarDistributionConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.optimizer import OptimizerConfig
from pfns.priors.prior import AdhocPriorConfig
from pfns.train import MainConfig
from pfns.utils import default_device

from pfns.priors import multitask_regression


@dataclass(frozen=True)
class MultitaskTrainingPlan:
    """Configuration helper for multitask PFN experiments.

    The defaults intentionally mirror the settings used in the multitask PFN paper
    while keeping the values small enough for unit tests and quick smoke runs.
    """

    num_tasks: int = 5
    batch_size: int = 8
    seq_len: int = 64
    single_eval_pos: int = 48
    num_features: int = 8
    steps_per_epoch: int = 25
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.0
    emsize: int = 128
    nlayers: int = 4
    nhead: int = 4
    nhid: int = 512
    features_per_group: int = 1
    attention_between_features: bool = True
    use_task_summary_projection: bool = True
    train_mixed_precision: bool = False
    device: str = default_device

    # Prior hyperparameters
    weight_std: float = 1.0
    task_offset_std: float = 0.5
    bias_std: float = 0.1
    observation_noise: float = 0.05

    # Histogram estimation for the bar distribution criterion
    num_histogram_buckets: int = 32
    sample_batches_for_borders: int = 16

    # Optional overrides
    model_extra_args: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.seq_len <= self.single_eval_pos:
            raise ValueError(
                "seq_len must be strictly larger than single_eval_pos so that a test "
                "segment exists."
            )
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")
        if self.seq_len <= 0 or self.single_eval_pos <= 0:
            raise ValueError("seq_len and single_eval_pos must be positive.")
        if self.num_histogram_buckets < 2:
            raise ValueError("num_histogram_buckets must be at least 2.")
        if self.sample_batches_for_borders <= 0:
            raise ValueError("sample_batches_for_borders must be positive.")
        if self.nhead <= 0 or self.emsize % self.nhead != 0:
            raise ValueError("nhead must divide emsize.")

    def spawn_for_tasks(self, num_tasks: int) -> MultitaskTrainingPlan:
        """Return a copy of the plan with a different number of tasks."""

        return replace(self, num_tasks=num_tasks)

    def prior_kwargs(self) -> dict:
        """Keyword arguments for the multitask prior helper."""

        return {
            "num_tasks": self.num_tasks,
            "weight_std": self.weight_std,
            "task_offset_std": self.task_offset_std,
            "bias_std": self.bias_std,
            "observation_noise": self.observation_noise,
            "device": self.device,
        }


def _multitask_prior_kwargs(plan: MultitaskTrainingPlan) -> dict:
    return plan.prior_kwargs()


def estimate_target_borders(plan: MultitaskTrainingPlan) -> Sequence[float]:
    """Estimate target value borders for the bar distribution criterion."""

    get_batch = multitask_regression.get_batch
    collected: list[torch.Tensor] = []

    for sample_idx in range(plan.sample_batches_for_borders):
        batch = get_batch(
            batch_size=plan.batch_size,
            seq_len=plan.seq_len,
            num_features=plan.num_features,
            single_eval_pos=plan.single_eval_pos,
            **_multitask_prior_kwargs(plan),
        )
        test_targets = batch.target_y[:, plan.single_eval_pos :]
        if test_targets.numel() == 0:
            continue
        collected.append(test_targets.reshape(-1))
        if collected and collected[-1].isnan().any():
            raise ValueError("Encountered NaNs while sampling multitask targets.")

    if not collected:
        raise RuntimeError("Unable to collect any targets for estimating borders.")

    concatenated = torch.cat(collected)
    quantiles = torch.linspace(0.0, 1.0, plan.num_histogram_buckets + 1, device=concatenated.device)
    borders = torch.quantile(concatenated, quantiles, interpolation="linear")
    borders = torch.unique_consecutive(borders)

    if borders.numel() < 2:
        # Fallback to a symmetric interval around the observed mean
        mean = concatenated.mean().item()
        std = concatenated.std(unbiased=False).item() or 1.0
        half_width = max(1.0, 3.0 * std)
        borders = torch.tensor([mean - half_width, mean + half_width], device=concatenated.device)

    if not torch.isfinite(borders).all():
        raise ValueError("Non-finite values encountered while estimating borders.")

    return borders.cpu().tolist()


def build_multitask_main_config(plan: MultitaskTrainingPlan) -> MainConfig:
    """Create a `MainConfig` tailored for multitask hierarchical attention."""

    borders = estimate_target_borders(plan)

    prior_config = AdhocPriorConfig(
        prior_names=["multitask_regression"],
        prior_kwargs=_multitask_prior_kwargs(plan),
    )

    criterion_config = BarDistributionConfig(borders=borders, full_support=True)

    transformer_config = TransformerConfig(
        criterion=criterion_config,
        emsize=plan.emsize,
        nhid=plan.nhid,
        nlayers=plan.nlayers,
        nhead=plan.nhead,
        features_per_group=plan.features_per_group,
        attention_between_features=plan.attention_between_features,
        use_hierarchical_attention=True,
        use_task_summary_projection=plan.use_task_summary_projection,
        model_extra_args=plan.model_extra_args,
    )

    batch_sampler = BatchShapeSamplerConfig(
        batch_size=plan.batch_size,
        min_single_eval_pos=plan.single_eval_pos,
        max_seq_len=plan.seq_len + 1,
        min_num_features=plan.num_features,
        max_num_features=plan.num_features,
        fixed_num_test_instances=plan.seq_len - plan.single_eval_pos,
        min_num_tasks=plan.num_tasks,
        max_num_tasks=plan.num_tasks,
    )

    optimizer_config = OptimizerConfig(
        optimizer="adamw",
        lr=plan.lr,
        weight_decay=plan.weight_decay,
    )

    return MainConfig(
        priors=[prior_config],
        optimizer=optimizer_config,
        model=transformer_config,
        batch_shape_sampler=batch_sampler,
        epochs=plan.epochs,
        steps_per_epoch=plan.steps_per_epoch,
        train_mixed_precision=plan.train_mixed_precision,
        scheduler="constant",
        warmup_epochs=max(1, math.ceil(plan.epochs / 10)),
        n_targets_per_input=1,
        num_workers=0,
        progress_bar=False,
    )


__all__ = [
    "MultitaskTrainingPlan",
    "estimate_target_borders",
    "build_multitask_main_config",
]
