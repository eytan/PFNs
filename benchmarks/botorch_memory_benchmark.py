#!/usr/bin/env python3
"""Memory benchmark for PFN KV caching with BoTorch optimization.

This script compares memory usage and result consistency between:
1. Standard BoTorch acquisition function optimization (O(b × n) training data)
2. Cached training context optimization (O(n) training data)

Uses a randomly initialized TableTransformer model with BarDistribution.

Usage:
    python benchmarks/botorch_memory_benchmark.py
"""

import gc
import time

import torch
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from pfns.botorch_pfn import PFNModel
from pfns.model.bar_distribution import BarDistribution
from pfns.model.transformer import TableTransformer


def get_memory_mb(device: str) -> float:
    """Get current memory allocated in MB."""
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_memory_mb(device: str) -> float:
    """Get peak memory allocated in MB."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory_stats(device: str) -> None:
    """Reset memory stats."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def create_random_pfn_model(n_dims: int, device: str):
    """Create a randomly initialized TableTransformer with BarDistribution."""
    # BarDistribution with 101 borders = 100 buckets
    num_buckets = 100
    borders = torch.linspace(-3, 3, num_buckets + 1)

    # Create transformer model with proper decoder for BarDistribution
    model = TableTransformer(
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=4,
        batch_first=True,
        cache_trainset_representation=False,
        decoder_dict={"standard": (None, num_buckets)},  # Output num_buckets logits
    )

    # Add BarDistribution criterion for regression
    model.criterion = BarDistribution(borders=borders)

    model = model.to(device)
    model.eval()

    return model


def run_botorch_benchmark(
    n_train: int = 50,
    n_dims: int = 5,
    num_restarts: int = 32,
    raw_samples: int = 128,
    n_trials: int = 3,
):
    """Run BoTorch acquisition optimization benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"BoTorch Acquisition Optimization Memory Benchmark")
    print(f"{'='*70}")
    print(f"\nDevice: {device}")
    print(f"Configuration:")
    print(f"  Training points (n): {n_train}")
    print(f"  Input dimensions (d): {n_dims}")
    print(f"  Num restarts (b): {num_restarts}")
    print(f"  Raw samples: {raw_samples}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Expected memory ratio for train data: {num_restarts}x")

    # Create randomly initialized model
    print(f"\nCreating randomly initialized TableTransformer...")
    pfn_model = create_random_pfn_model(n_dims, device)
    print(f"  Model created: {sum(p.numel() for p in pfn_model.parameters())} params")

    # Generate synthetic training data
    torch.manual_seed(42)
    train_X = torch.rand(n_train, n_dims, device=device)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) + 0.1 * torch.randn(
        n_train, 1, device=device
    )
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()
    bounds = torch.stack([torch.zeros(n_dims), torch.ones(n_dims)]).to(device)

    # Create PFNModel wrapper
    model = PFNModel(
        train_X=train_X,
        train_Y=train_Y,
        model=pfn_model,
        batch_first=True,
    )

    best_f = train_Y.max().item()

    # Storage for results
    standard_results = []
    cached_results = []
    standard_times = []
    cached_times = []

    print(f"\n{'='*70}")
    print("Running benchmarks...")
    print(f"{'='*70}")

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Reset random state for fair comparison
        torch.manual_seed(42 + trial)

        # Benchmark 1: Standard optimization (no caching)
        print("\n[Standard] Running optimize_acqf without caching...")
        reset_memory_stats(device)
        start_time = time.time()

        acqf_standard = ExpectedImprovement(model, best_f=best_f)
        candidates_standard, value_standard = optimize_acqf(
            acqf_standard,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": num_restarts},
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        standard_time = time.time() - start_time
        standard_peak = get_peak_memory_mb(device)
        standard_times.append(standard_time)

        standard_results.append(
            {
                "candidates": candidates_standard.detach().cpu(),
                "value": value_standard.item(),
                "peak_mem": standard_peak,
            }
        )

        print(f"  Time: {standard_time:.3f}s")
        if device.startswith("cuda"):
            print(f"  Peak memory: {standard_peak:.1f} MB")
        print(f"  Best value: {value_standard.item():.6f}")
        print(f"  Candidate: {candidates_standard.squeeze().cpu().numpy()}")

        # Reset random state for fair comparison
        torch.manual_seed(42 + trial)

        # Benchmark 2: Cached optimization
        print("\n[Cached] Running optimize_acqf with caching...")
        reset_memory_stats(device)
        start_time = time.time()

        with model.cache_training_context():
            acqf_cached = ExpectedImprovement(model, best_f=best_f)
            candidates_cached, value_cached = optimize_acqf(
                acqf_cached,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": num_restarts},
            )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        cached_time = time.time() - start_time
        cached_peak = get_peak_memory_mb(device)
        cached_times.append(cached_time)

        cached_results.append(
            {
                "candidates": candidates_cached.detach().cpu(),
                "value": value_cached.item(),
                "peak_mem": cached_peak,
            }
        )

        print(f"  Time: {cached_time:.3f}s")
        if device.startswith("cuda"):
            print(f"  Peak memory: {cached_peak:.1f} MB")
        print(f"  Best value: {value_cached.item():.6f}")
        print(f"  Candidate: {candidates_cached.squeeze().cpu().numpy()}")

        # Compare this trial
        candidate_diff = (candidates_standard - candidates_cached).abs().max().item()
        value_diff = abs(value_standard.item() - value_cached.item())
        print(f"\n  Trial comparison:")
        print(f"    Max candidate diff: {candidate_diff:.6f}")
        print(f"    Value diff: {value_diff:.6f}")
        print(f"    Speedup: {standard_time / cached_time:.2f}x")

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_standard_time = sum(standard_times) / len(standard_times)
    avg_cached_time = sum(cached_times) / len(cached_times)

    print(f"\nTiming (averaged over {n_trials} trials):")
    print(f"  Standard: {avg_standard_time:.3f}s")
    print(f"  Cached:   {avg_cached_time:.3f}s")
    print(f"  Speedup:  {avg_standard_time / avg_cached_time:.2f}x")

    if device.startswith("cuda"):
        avg_standard_mem = sum(r["peak_mem"] for r in standard_results) / len(
            standard_results
        )
        avg_cached_mem = sum(r["peak_mem"] for r in cached_results) / len(
            cached_results
        )
        print(f"\nMemory (averaged over {n_trials} trials):")
        print(f"  Standard peak: {avg_standard_mem:.1f} MB")
        print(f"  Cached peak:   {avg_cached_mem:.1f} MB")
        print(f"  Reduction:     {(1 - avg_cached_mem / avg_standard_mem) * 100:.1f}%")

    # Check result consistency
    print(f"\nResult Consistency Check:")
    all_match = True
    for i in range(n_trials):
        std_cand = standard_results[i]["candidates"]
        cached_cand = cached_results[i]["candidates"]
        std_val = standard_results[i]["value"]
        cached_val = cached_results[i]["value"]

        cand_diff = (std_cand - cached_cand).abs().max().item()
        val_diff = abs(std_val - cached_val)

        match = cand_diff < 0.01 and val_diff < 0.01
        status = "✓" if match else "✗"
        print(
            f"  Trial {i+1}: {status} candidate_diff={cand_diff:.6f}, value_diff={val_diff:.6f}"
        )
        all_match = all_match and match

    if all_match:
        print("\n✓ All trials produced consistent results between cached and standard!")
    else:
        print(
            "\n⚠ Some differences detected - this may be due to optimization stochasticity"
        )
        print(
            "  (Different code paths may lead to slightly different floating point results)"
        )

    return standard_results, cached_results


def verify_prediction_consistency():
    """Verify that cached and non-cached predictions are numerically identical."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print("Direct Prediction Consistency Verification")
    print(f"{'='*70}")

    # Create model
    n_dims = 5
    n_train = 50
    n_test = 10

    pfn_model = create_random_pfn_model(n_dims, device)

    torch.manual_seed(42)
    train_X = torch.rand(n_train, n_dims, device=device)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) + 0.1 * torch.randn(
        n_train, 1, device=device
    )
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()

    # Create test points
    test_X = torch.rand(n_test, n_dims, device=device)

    # Create PFNModel wrapper
    model = PFNModel(
        train_X=train_X,
        train_Y=train_Y,
        model=pfn_model,
        batch_first=True,
    )

    # Get standard predictions
    print("\nComputing standard predictions...")
    with torch.no_grad():
        posterior_std = model.posterior(test_X)
        mean_std = posterior_std.mean
        var_std = posterior_std.variance

    # Get cached predictions
    print("Computing cached predictions...")
    with model.cache_training_context():
        with torch.no_grad():
            posterior_cached = model.posterior(test_X)
            mean_cached = posterior_cached.mean
            var_cached = posterior_cached.variance

    # Compare
    mean_diff = (mean_std - mean_cached).abs().max().item()
    var_diff = (var_std - var_cached).abs().max().item()

    print(f"\nResults:")
    print(f"  Max mean difference: {mean_diff:.2e}")
    print(f"  Max variance difference: {var_diff:.2e}")

    if mean_diff < 1e-5 and var_diff < 1e-5:
        print("\n✓ Cached and standard predictions are numerically identical!")
        return True
    else:
        print("\n✗ Predictions differ significantly - there may be a bug")
        return False


if __name__ == "__main__":
    verify_prediction_consistency()
    run_botorch_benchmark()
