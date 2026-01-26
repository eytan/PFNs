"""Tests for PFN caching functionality.

Tests verify:
1. Gradient flow through test_x when using cached training context
2. Output consistency between cached and non-cached modes
3. Batch broadcast behavior with cached KV
4. Sequential vs batch evaluation equivalence
"""

import pytest
import torch

from pfns.model.transformer import TableTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing PFN caching on {device=}.")

dtype = torch.float32  # Use float32 for gradient tests


@pytest.fixture
def minimal_transformer():
    """Create a minimal TableTransformer for testing."""
    model = TableTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        batch_first=True,
        cache_trainset_representation=False,
    )
    model = model.to(device=device, dtype=dtype)
    # Add small non-zero values to weights to make outputs more interesting
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    return model


@pytest.fixture
def sample_data():
    """Generate sample training and test data."""
    n_train = 10
    n_test = 5
    n_features = 4
    batch_size = 1  # Base batch size for training data

    train_x = torch.randn(batch_size, n_train, n_features, device=device, dtype=dtype)
    train_y = torch.randn(batch_size, n_train, 1, device=device, dtype=dtype)
    test_x = torch.randn(batch_size, n_test, n_features, device=device, dtype=dtype)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
    }


def test_cache_output_consistency(minimal_transformer, sample_data):
    """Test that cached and non-cached outputs are identical."""
    model = minimal_transformer
    train_x = sample_data["train_x"]
    train_y = sample_data["train_y"]
    test_x = sample_data["test_x"]

    # Non-cached forward pass
    model.cache_trainset_representation = False
    with torch.no_grad():
        output_no_cache = model(x=train_x, y=train_y, test_x=test_x)

    # Cached forward pass: first populate cache, then use it
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    with torch.no_grad():
        # Populate cache with training data
        _ = model(x=train_x, y=train_y, test_x=test_x[:, :1, :])  # dummy test

        # Use cache for actual test
        output_with_cache = model(x=None, y=None, test_x=test_x)

    # Clear cache
    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    assert torch.allclose(output_no_cache, output_with_cache, atol=1e-7), (
        f"Cached and non-cached outputs differ. "
        f"Max diff: {(output_no_cache - output_with_cache).abs().max()}"
    )


def test_cache_gradient_flow(minimal_transformer, sample_data):
    """Test that gradients flow through test_x when using cached training context.

    This is critical for acquisition function optimization in BoTorch.
    """
    model = minimal_transformer
    train_x = sample_data["train_x"]
    train_y = sample_data["train_y"]
    test_x = sample_data["test_x"].clone().requires_grad_(True)

    # Enable caching
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    # Populate cache with no gradients on training data
    with torch.no_grad():
        _ = model(x=train_x, y=train_y, test_x=torch.zeros_like(test_x[:, :1, :]))

    # Forward pass using cache - should allow gradients on test_x
    output = model(x=None, y=None, test_x=test_x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Verify gradients exist on test_x
    assert test_x.grad is not None, "test_x.grad should not be None"
    assert not torch.all(test_x.grad == 0), "test_x.grad should not be all zeros"

    # Cleanup
    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False


def test_cache_gradient_correctness(minimal_transformer, sample_data):
    """Test that gradients are identical with and without caching."""
    model = minimal_transformer
    train_x = sample_data["train_x"]
    train_y = sample_data["train_y"]

    # Test 1: Compute gradients without caching
    test_x_no_cache = sample_data["test_x"].clone().requires_grad_(True)
    model.cache_trainset_representation = False
    output_no_cache = model(x=train_x, y=train_y, test_x=test_x_no_cache)
    output_no_cache.sum().backward()
    grad_no_cache = test_x_no_cache.grad.clone()

    # Test 2: Compute gradients with caching
    test_x_with_cache = sample_data["test_x"].clone().requires_grad_(True)
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    # Populate cache
    with torch.no_grad():
        _ = model(
            x=train_x, y=train_y, test_x=torch.zeros_like(test_x_with_cache[:, :1, :])
        )

    # Forward with cache
    output_with_cache = model(x=None, y=None, test_x=test_x_with_cache)
    output_with_cache.sum().backward()
    grad_with_cache = test_x_with_cache.grad.clone()

    # Cleanup
    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    # Compare gradients
    assert torch.allclose(grad_no_cache, grad_with_cache, atol=1e-7), (
        f"Gradients differ. Max diff: {(grad_no_cache - grad_with_cache).abs().max()}"
    )


def test_cache_batch_broadcast(minimal_transformer, sample_data):
    """Test that cached KV with batch_size=1 can be used with larger query batches.

    This is essential for efficient acquisition function optimization where
    we cache training context once and evaluate many test points in parallel.
    """
    model = minimal_transformer
    train_x = sample_data["train_x"]  # (1, n, d)
    train_y = sample_data["train_y"]  # (1, n, 1)
    n_features = sample_data["n_features"]

    # Create a larger batch of test points
    batch_size = 4
    n_test = 3
    test_x_batch = torch.randn(
        batch_size, n_test, n_features, device=device, dtype=dtype
    )

    # Compute reference: sequential evaluation (one at a time)
    model.cache_trainset_representation = False
    reference_outputs = []
    for i in range(batch_size):
        with torch.no_grad():
            out = model(x=train_x, y=train_y, test_x=test_x_batch[i : i + 1])
        reference_outputs.append(out)
    reference_output = torch.cat(reference_outputs, dim=0)

    # Enable caching with batch_size=1
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    # Populate cache with training data (batch_size=1)
    with torch.no_grad():
        _ = model(
            x=train_x,
            y=train_y,
            test_x=torch.zeros(1, 1, n_features, device=device, dtype=dtype),
        )

    # Evaluate all test points in parallel using cache
    # The test_x has batch_size=4, but cache has batch_size=1
    # This requires broadcasting in the attention layer
    with torch.no_grad():
        # Expand train context for batch
        batch_output = model(x=None, y=None, test_x=test_x_batch)

    # Cleanup
    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    # Compare outputs
    assert batch_output.shape == reference_output.shape, (
        f"Shape mismatch: batch={batch_output.shape}, ref={reference_output.shape}"
    )
    assert torch.allclose(reference_output, batch_output, atol=1e-7), (
        f"Batch output differs from sequential. "
        f"Max diff: {(reference_output - batch_output).abs().max()}"
    )


def test_attention_caching():
    """Test MultiHeadAttention caching produces identical outputs and correct gradients."""
    from pfns.model.multi_head_attention import MultiHeadAttention

    embed_dim = 128
    nhead = 4
    n_seq_kv = 316
    n_seq_q = 534
    n_batch = 7

    att = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
    )

    x_kv = torch.randn(n_batch, n_seq_kv, embed_dim, device=device, dtype=dtype)
    x_q = torch.randn(n_batch, n_seq_q, embed_dim, device=device, dtype=dtype)

    # Test output consistency between cached and non-cached
    y = att(x_q, x_kv, cache_kv=True)
    y_ = att(x_q, use_cached_kv=True)
    assert torch.allclose(y, y_, atol=1e-7), (
        f"Cached and non-cached attention outputs differ. "
        f"Max diff: {(y - y_).abs().max()}"
    )

    # Gradients should fail for train part (caching requires no grad)
    x_q_grad = x_q.clone().requires_grad_(True)
    with pytest.raises(AssertionError):
        att(x_q_grad, x_kv, cache_kv=True)

    # Gradients should flow through test part when using cache
    x_q_grad = x_q.clone().requires_grad_(True)
    y_ = att(x_q_grad, x_kv, use_cached_kv=True)
    y_.mean().backward()
    assert x_q_grad.grad is not None
    grads_with_cache = x_q_grad.grad.clone()

    # Compute gradients without caching - should match
    x_q_grad = x_q.clone().requires_grad_(True)
    y_ = att(x_q_grad, x_kv)
    y_.mean().backward()
    assert x_q_grad.grad is not None
    grads_without_cache = x_q_grad.grad.clone()

    assert torch.allclose(grads_with_cache, grads_without_cache, atol=1e-7), (
        f"Attention gradients differ. "
        f"Max diff: {(grads_with_cache - grads_without_cache).abs().max()}"
    )

    # Cleanup
    att.empty_kv_cache()


def test_attention_cache_batch_broadcast():
    """Test MultiHeadAttention layer directly for batch broadcast with cached KV."""
    from pfns.model.multi_head_attention import MultiHeadAttention

    embed_dim = 32
    nhead = 2
    n_seq_kv = 10
    n_seq_q = 5

    att = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
    )

    # KV with batch_size=1
    x_kv = torch.randn(1, n_seq_kv, embed_dim, device=device, dtype=dtype)

    # Query with batch_size=4
    batch_size_q = 4
    x_q = torch.randn(batch_size_q, n_seq_q, embed_dim, device=device, dtype=dtype)

    # Cache KV with batch_size=1
    with torch.no_grad():
        att(x_kv, x_kv, cache_kv=True)

    # Reference: compute each batch element separately by expanding KV
    x_kv_expanded = x_kv.expand(batch_size_q, -1, -1)
    with torch.no_grad():
        ref_output = att(x_q, x_kv_expanded)

    # Test using cache with batch_size=4 query
    # Batch broadcast is supported - verify it works correctly
    with torch.no_grad():
        cached_output = att(x_q, use_cached_kv=True)

    assert cached_output.shape == ref_output.shape
    assert torch.allclose(ref_output, cached_output, atol=1e-7), (
        f"Cached attention output differs. "
        f"Max diff: {(ref_output - cached_output).abs().max()}"
    )

    # Cleanup
    att.empty_kv_cache()


def test_deterministic_seeded_outputs(minimal_transformer, sample_data):
    """Test that with same seed, outputs are deterministic."""
    model = minimal_transformer
    train_x = sample_data["train_x"]
    train_y = sample_data["train_y"]
    test_x = sample_data["test_x"]

    torch.manual_seed(42)
    model.cache_trainset_representation = False
    with torch.no_grad():
        output1 = model(x=train_x, y=train_y, test_x=test_x)

    torch.manual_seed(42)
    with torch.no_grad():
        output2 = model(x=train_x, y=train_y, test_x=test_x)

    assert torch.allclose(output1, output2, atol=1e-7), "Outputs should be deterministic"


def test_cache_invalidation_on_data_change(minimal_transformer, sample_data):
    """Test that cache is properly invalidated when training data changes.

    In a real BO loop, training data grows as new observations are added.
    The cache must be invalidated when this happens to avoid stale predictions.
    """
    model = minimal_transformer
    train_x = sample_data["train_x"].clone()
    train_y = sample_data["train_y"].clone()
    test_x = sample_data["test_x"]
    n_features = sample_data["n_features"]

    # Step 1: Populate cache with initial training data
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    with torch.no_grad():
        _ = model(
            x=train_x,
            y=train_y,
            test_x=torch.zeros(1, 1, n_features, device=device, dtype=dtype),
        )

    # Get prediction with original cache
    with torch.no_grad():
        output_original = model(x=None, y=None, test_x=test_x)

    # Step 2: Clear cache and repopulate with DIFFERENT training data
    model.empty_trainset_representation_cache()

    # Create modified training data (add noise)
    train_x_modified = train_x + 0.5 * torch.randn_like(train_x)
    train_y_modified = train_y + 0.5 * torch.randn_like(train_y)

    with torch.no_grad():
        _ = model(
            x=train_x_modified,
            y=train_y_modified,
            test_x=torch.zeros(1, 1, n_features, device=device, dtype=dtype),
        )

    # Get prediction with new cache
    with torch.no_grad():
        output_modified = model(x=None, y=None, test_x=test_x)

    # Step 3: Verify outputs are DIFFERENT (cache was properly updated)
    assert not torch.allclose(output_original, output_modified, atol=1e-3), (
        "Outputs should differ after training data change. "
        "Cache may not have been properly invalidated/repopulated."
    )

    # Step 4: Verify the new output matches a fresh non-cached computation
    model.cache_trainset_representation = False
    model.empty_trainset_representation_cache()

    with torch.no_grad():
        output_fresh = model(x=train_x_modified, y=train_y_modified, test_x=test_x)

    assert torch.allclose(output_modified, output_fresh, atol=1e-7), (
        f"Cached output with new data should match fresh computation. "
        f"Max diff: {(output_modified - output_fresh).abs().max()}"
    )


def test_cache_with_batch_first_false():
    """Test caching works correctly with batch_first=False configuration."""
    model = TableTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        batch_first=False,  # Key difference from other tests
        cache_trainset_representation=False,
    )
    model = model.to(device=device, dtype=dtype)

    # Add small non-zero values to weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))

    n_train, n_test, n_features = 10, 5, 4
    batch_size = 1

    # For batch_first=False, shape is (seq, batch, features)
    train_x = torch.randn(n_train, batch_size, n_features, device=device, dtype=dtype)
    train_y = torch.randn(n_train, batch_size, 1, device=device, dtype=dtype)
    test_x = torch.randn(n_test, batch_size, n_features, device=device, dtype=dtype)

    # Non-cached forward pass
    model.cache_trainset_representation = False
    with torch.no_grad():
        output_no_cache = model(x=train_x, y=train_y, test_x=test_x)

    # Cached forward pass
    model.cache_trainset_representation = True
    model.empty_trainset_representation_cache()

    with torch.no_grad():
        # Populate cache
        _ = model(
            x=train_x,
            y=train_y,
            test_x=torch.zeros(1, batch_size, n_features, device=device, dtype=dtype),
        )

        # Use cache
        output_with_cache = model(x=None, y=None, test_x=test_x)

    # Cleanup
    model.empty_trainset_representation_cache()
    model.cache_trainset_representation = False

    assert torch.allclose(output_no_cache, output_with_cache, atol=1e-7), (
        f"batch_first=False: Cached and non-cached outputs differ. "
        f"Max diff: {(output_no_cache - output_with_cache).abs().max()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
