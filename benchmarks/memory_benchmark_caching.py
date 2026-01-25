#!/usr/bin/env python3
"""Memory benchmark for PFN KV caching optimization.

This script compares memory usage between:
1. Standard BoTorch acquisition function optimization (O(b × n) training data)
2. Cached training context optimization (O(n) training data)

Usage:
    python benchmarks/memory_benchmark_caching.py

Requirements:
    - torch
    - botorch
    - pfns4bo (for model download)
"""

import gc
import time
from contextlib import contextmanager
from typing import Callable

import torch

# Memory tracking utilities
def get_memory_allocated(device: str) -> float:
    """Get current memory allocated in MB."""
    if device == "cpu":
        return 0.0  # Can't track CPU memory easily
    elif device.startswith("cuda"):
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif device.startswith("mps"):
        # MPS doesn't have built-in memory tracking
        return 0.0
    return 0.0


def get_peak_memory(device: str) -> float:
    """Get peak memory allocated in MB."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_peak_memory(device: str) -> None:
    """Reset peak memory counter."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_memory(device: str, label: str):
    """Context manager to track memory usage."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    reset_peak_memory(device)
    start_mem = get_memory_allocated(device)
    start_time = time.time()

    yield

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    end_time = time.time()
    peak_mem = get_peak_memory(device)
    end_mem = get_memory_allocated(device)

    print(f"\n{label}:")
    print(f"  Duration: {end_time - start_time:.3f}s")
    if device.startswith("cuda"):
        print(f"  Memory before: {start_mem:.1f} MB")
        print(f"  Memory after: {end_mem:.1f} MB")
        print(f"  Peak memory: {peak_mem:.1f} MB")


def run_benchmark_with_model():
    """Run benchmark using actual PFN model with BoTorch."""
    try:
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from pfns.botorch_pfn import PFNModel
    except ImportError as e:
        print(f"Could not import BoTorch components: {e}")
        print("Install with: pip install botorch pfns4bo")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Running BoTorch Memory Benchmark on {device}")
    print(f"{'='*60}")

    # Configuration
    n_train = 50  # Number of training points
    n_dims = 5    # Input dimensionality
    num_restarts = 32  # Acquisition function restarts (batch size)
    raw_samples = 128  # Raw samples for initialization

    print(f"\nConfiguration:")
    print(f"  Training points (n): {n_train}")
    print(f"  Input dimensions (d): {n_dims}")
    print(f"  Num restarts (b): {num_restarts}")
    print(f"  Raw samples: {raw_samples}")
    print(f"  Expected memory ratio: O(b×n) / O(n) = {num_restarts}x")

    # Generate synthetic training data
    torch.manual_seed(42)
    train_X = torch.rand(n_train, n_dims, device=device)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) + 0.1 * torch.randn(n_train, 1, device=device)
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()  # Normalize for model
    bounds = torch.stack([torch.zeros(n_dims), torch.ones(n_dims)]).to(device)

    print("\nLoading PFN model...")
    try:
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            batch_first=True,
            load_training_checkpoint=False,
        )
    except Exception as e:
        print(f"Could not load PFN model: {e}")
        print("Trying to create a simple model for testing...")
        return

    best_f = train_Y.max().item()

    # Benchmark 1: Standard acquisition optimization (no caching)
    print("\n" + "-"*40)
    print("Benchmark 1: Standard optimization (no caching)")
    print("-"*40)

    with track_memory(device, "Standard optimize_acqf"):
        acqf = ExpectedImprovement(model, best_f=best_f)
        candidates_standard, value_standard = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": num_restarts},
        )

    print(f"  Best value found: {value_standard.item():.4f}")
    print(f"  Best candidate: {candidates_standard.squeeze().cpu().numpy()}")

    # Benchmark 2: Cached training context optimization
    print("\n" + "-"*40)
    print("Benchmark 2: Cached training context optimization")
    print("-"*40)

    with track_memory(device, "Cached optimize_acqf"):
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

    print(f"  Best value found: {value_cached.item():.4f}")
    print(f"  Best candidate: {candidates_cached.squeeze().cpu().numpy()}")

    # Compare results
    print("\n" + "="*40)
    print("Result Comparison")
    print("="*40)
    candidate_diff = (candidates_standard - candidates_cached).abs().max().item()
    value_diff = abs(value_standard.item() - value_cached.item())
    print(f"Max candidate difference: {candidate_diff:.6f}")
    print(f"Value difference: {value_diff:.6f}")

    if candidate_diff < 1e-3 and value_diff < 1e-3:
        print("✓ Results are numerically equivalent!")
    else:
        print("⚠ Results differ - this may be expected due to optimization stochasticity")


def run_benchmark_with_transformer():
    """Run benchmark using TableTransformer directly (no BoTorch dependency)."""
    from pfns.model.transformer import TableTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"\n{'='*60}")
    print(f"Running Transformer Memory Benchmark on {device}")
    print(f"{'='*60}")

    # Configuration
    n_train = 100  # Number of training points
    n_test = 1     # Number of test points per batch
    n_features = 10  # Input dimensionality
    batch_size = 64  # Batch size for parallel evaluation
    n_iterations = 10  # Number of forward passes

    # Model config
    ninp = 64
    nhead = 4
    nhid = 128
    nlayers = 4

    print(f"\nConfiguration:")
    print(f"  Training points (n): {n_train}")
    print(f"  Test points per batch (q): {n_test}")
    print(f"  Features (d): {n_features}")
    print(f"  Batch size (b): {batch_size}")
    print(f"  Model: ninp={ninp}, nhead={nhead}, nhid={nhid}, nlayers={nlayers}")
    print(f"  Expected memory ratio: O(b×n) / O(n) ≈ {batch_size}x for training data")

    # Create model
    torch.manual_seed(42)
    model = TableTransformer(
        ninp=ninp,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        batch_first=True,
        cache_trainset_representation=False,
    ).to(device=device, dtype=dtype)
    model.eval()

    # Generate synthetic data
    train_x = torch.randn(1, n_train, n_features, device=device, dtype=dtype)
    train_y = torch.randn(1, n_train, 1, device=device, dtype=dtype)

    # Benchmark 1: Standard forward passes with replicated training data
    print("\n" + "-"*40)
    print("Benchmark 1: Standard forward (O(b × n) training data)")
    print("-"*40)

    with track_memory(device, "Standard forward"):
        for _ in range(n_iterations):
            # Replicate training data for batch
            train_x_batch = train_x.expand(batch_size, -1, -1)
            train_y_batch = train_y.expand(batch_size, -1, -1)
            test_x = torch.randn(batch_size, n_test, n_features, device=device, dtype=dtype)

            with torch.no_grad():
                output = model(x=train_x_batch, y=train_y_batch, test_x=test_x)

    # Benchmark 2: Cached training context
    print("\n" + "-"*40)
    print("Benchmark 2: Cached forward (O(n) training data)")
    print("-"*40)

    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    with track_memory(device, "Cached forward"):
        # Populate cache once
        with torch.no_grad():
            dummy_test = torch.zeros(1, 1, n_features, device=device, dtype=dtype)
            _ = model(x=train_x, y=train_y, test_x=dummy_test)

        # Now run iterations using cache
        for _ in range(n_iterations):
            test_x = torch.randn(batch_size, n_test, n_features, device=device, dtype=dtype)
            with torch.no_grad():
                output = model(x=None, y=None, test_x=test_x)

    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    # Benchmark 3: Verify gradient flow
    print("\n" + "-"*40)
    print("Benchmark 3: Gradient verification")
    print("-"*40)

    # Standard gradient
    model.cache_trainset_representation = False
    test_x_std = torch.randn(1, n_test, n_features, device=device, dtype=dtype, requires_grad=True)
    output_std = model(x=train_x, y=train_y, test_x=test_x_std)
    output_std.sum().backward()
    grad_std = test_x_std.grad.clone()

    # Cached gradient
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()
    with torch.no_grad():
        _ = model(x=train_x, y=train_y, test_x=torch.zeros_like(test_x_std[:, :1, :]))

    test_x_cached = test_x_std.detach().clone().requires_grad_(True)
    output_cached = model(x=None, y=None, test_x=test_x_cached)
    output_cached.sum().backward()
    grad_cached = test_x_cached.grad.clone()

    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    # Compare gradients
    grad_diff = (grad_std - grad_cached).abs().max().item()
    output_diff = (output_std - output_cached).abs().max().item()

    print(f"  Max gradient difference: {grad_diff:.2e}")
    print(f"  Max output difference: {output_diff:.2e}")

    if grad_diff < 1e-4 and output_diff < 1e-4:
        print("  ✓ Gradients and outputs match!")
    else:
        print("  ⚠ Differences detected - check implementation")


def main():
    print("PFN KV Caching Memory Benchmark")
    print("================================\n")

    # Run transformer-level benchmark (no external dependencies)
    run_benchmark_with_transformer()

    # Try to run BoTorch benchmark if available
    try:
        run_benchmark_with_model()
    except Exception as e:
        print(f"\nSkipped BoTorch benchmark: {e}")


if __name__ == "__main__":
    main()
