from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pfns.train import MainConfig, train as run_training_loop
from pfns.utils import default_device

from .configs import MultitaskTrainingPlan, build_multitask_main_config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a hierarchical multitask PFN using the built-in trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks per batch.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of task datasets per batch.")
    parser.add_argument("--seq-len", type=int, default=64, help="Total sequence length (train + test).")
    parser.add_argument("--single-eval-pos", type=int, default=48, help="Training sequence length inside each dataset.")
    parser.add_argument("--num-features", type=int, default=8, help="Number of features per task dataset.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--steps-per-epoch", type=int, default=25, help="Steps per epoch for the trainer.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument("--emsize", type=int, default=128, help="Embedding dimension of the transformer.")
    parser.add_argument("--nhid", type=int, default=512, help="Hidden size of the feed-forward network.")
    parser.add_argument("--nlayers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument(
        "--features-per-group",
        type=int,
        default=1,
        help="Number of features grouped together for column-wise attention.",
    )
    parser.add_argument(
        "--disable-feature-attention",
        action="store_true",
        help="Disable between-feature attention (reverts to per-feature processing).",
    )
    parser.add_argument(
        "--no-summary-projection",
        action="store_true",
        help="Disable the learnable projection on task summary tokens.",
    )
    parser.add_argument(
        "--histogram-buckets",
        type=int,
        default=32,
        help="Number of histogram buckets for the bar distribution criterion.",
    )
    parser.add_argument(
        "--border-samples",
        type=int,
        default=16,
        help="Number of synthetic batches to sample when estimating histogram borders.",
    )
    parser.add_argument(
        "--train-mixed-precision",
        action="store_true",
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for sampling and training (defaults to detected device).",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional path to save the generated config in YAML format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the config and exit without launching training.",
    )
    return parser


def plan_from_args(args: argparse.Namespace) -> MultitaskTrainingPlan:
    device = args.device or default_device
    return MultitaskTrainingPlan(
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        single_eval_pos=args.single_eval_pos,
        num_features=args.num_features,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        nhead=args.nhead,
        features_per_group=args.features_per_group,
        attention_between_features=not args.disable_feature_attention,
        use_task_summary_projection=not args.no_summary_projection,
        train_mixed_precision=args.train_mixed_precision,
        num_histogram_buckets=args.histogram_buckets,
        sample_batches_for_borders=args.border_samples,
        device=device,
    )


def run_from_plan(plan: MultitaskTrainingPlan, *, dry_run: bool = False) -> MainConfig:
    config = build_multitask_main_config(plan)
    if not dry_run:
        run_training_loop(config, device=plan.device)
    return config


def main(argv: Sequence[str] | None = None) -> MainConfig:
    parser = create_parser()
    args = parser.parse_args(argv)
    plan = plan_from_args(args)
    config = build_multitask_main_config(plan)

    if args.config_out is not None:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(config.to_yaml())
        print(f"Wrote multitask config to {args.config_out}")

    if args.dry_run:
        print("Dry run requested; skipping training.")
        return config

    run_training_loop(config, device=plan.device)
    return config


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
