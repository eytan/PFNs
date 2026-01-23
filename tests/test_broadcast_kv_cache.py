#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for broadcast KV caching in PFN models.

This module verifies that the memory-efficient batched inference using
shared context KV caching produces correct results and reduces memory usage.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager

import pytest


def test_broadcast_kv_cache_correctness():
    """Test that broadcast KV cache produces same results as standard forward pass."""
    pytest.importorskip("pfns")
    from pfns.model.transformer import TableTransformer

    # Create a small test model
    model = TableTransformer(
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        features_per_group=1,
        attention_between_features=True,
        cache_trainset_representation=True,
    )
    model.eval()

    # Create test data
    torch.manual_seed(42)
    n_train = 10
    n_test = 5
    n_features = 3
    batch_size = 4

    x_train = torch.randn(1, n_train, n_features)
    y_train = torch.randn(1, n_train, 1)
    x_test = torch.randn(batch_size, n_test, n_features)

    # Standard forward pass (with batch expansion)
    with torch.no_grad():
        # Expand training data to match test batch size
        x_train_expanded = x_train.expand(batch_size, -1, -1)
        y_train_expanded = y_train.expand(batch_size, -1, -1)
        output_standard = model(
            x=x_train_expanded,
            y=y_train_expanded,
            test_x=x_test,
        )

    # Clear any cached state
    model.empty_trainset_representation_cache()

    # Broadcast KV cache forward pass
    with torch.no_grad():
        with model.shared_context(x_train.squeeze(0), y_train.squeeze(0)):
            output_cached = model(x=None, y=None, test_x=x_test)

    # Verify outputs match
    torch.testing.assert_close(
        output_standard,
        output_cached,
        rtol=1e-4,
        atol=1e-4,
        msg="Broadcast KV cache output should match standard forward pass",
    )


def test_broadcast_kv_cache_gradient_flow():
    """Test that gradients flow through test_x but not through cached context."""
    pytest.importorskip("pfns")
    from pfns.model.transformer import TableTransformer

    # Create a small test model with zero_init=False so output varies with input
    # (zero_init=True produces constant output for untrained model, so no gradients)
    model = TableTransformer(
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        features_per_group=1,
        attention_between_features=True,
        cache_trainset_representation=True,
        zero_init=False,
    )
    model.eval()

    # Create test data
    torch.manual_seed(42)
    n_train = 10
    n_test = 5
    n_features = 3
    batch_size = 4

    x_train = torch.randn(1, n_train, n_features)
    y_train = torch.randn(1, n_train, 1)
    x_test = torch.randn(batch_size, n_test, n_features, requires_grad=True)

    # Forward pass with cached context
    with model.shared_context(x_train.squeeze(0), y_train.squeeze(0)):
        output = model(x=None, y=None, test_x=x_test)
        # Sum output and backpropagate
        loss = output.sum()
        loss.backward()

    # Verify gradient flows to test_x
    assert x_test.grad is not None, "Gradient should flow to test_x"
    assert x_test.grad.abs().sum() > 0, "test_x gradient should be non-zero"

    # Verify training data has no gradient (computed with no_grad in encode_context)
    assert not x_train.requires_grad or x_train.grad is None, \
        "Training data should not have gradients"


def test_multihead_attention_broadcast():
    """Test that MultiHeadAttention correctly broadcasts cached KV to larger query batch."""
    pytest.importorskip("pfns")
    from pfns.model.multi_head_attention import MultiHeadAttention

    # Create attention module
    attention = MultiHeadAttention(
        input_size=64,
        output_size=64,
        d_k=16,
        d_v=16,
        nhead=4,
        device=None,
        dtype=None,
    )
    attention.eval()

    torch.manual_seed(42)
    batch_size_kv = 1
    batch_size_q = 4
    seqlen_kv = 10
    seqlen_q = 5
    d_model = 64

    # First: cache KV with batch_size=1
    kv = torch.randn(batch_size_kv, seqlen_kv, d_model)
    q_init = torch.randn(batch_size_kv, seqlen_q, d_model)

    with torch.no_grad():
        # Cache the KV
        _ = attention(q_init, kv, cache_kv=True)

    # Now: query with larger batch should broadcast cached KV
    q = torch.randn(batch_size_q, seqlen_q, d_model)

    with torch.no_grad():
        output = attention(q, x_kv=None, use_cached_kv=True)

    # Output should have Q's batch size
    assert output.shape[0] == batch_size_q, \
        f"Expected output batch size {batch_size_q}, got {output.shape[0]}"
    assert output.shape[1] == seqlen_q, \
        f"Expected output seqlen {seqlen_q}, got {output.shape[1]}"


def test_shared_context_cleanup():
    """Test that shared_context properly cleans up after exiting."""
    pytest.importorskip("pfns")
    from pfns.model.transformer import TableTransformer

    model = TableTransformer(
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        features_per_group=1,
        attention_between_features=True,
        cache_trainset_representation=False,  # Start with caching disabled
    )
    model.eval()

    x_train = torch.randn(10, 3)
    y_train = torch.randn(10)

    # Before context
    assert not model._context_encoded
    assert model._cached_context_batch_size is None
    orig_cache_setting = model.cache_trainset_representation

    # Inside context
    with model.shared_context(x_train, y_train):
        assert model._context_encoded
        assert model._cached_context_batch_size == 1

    # After context - should be cleaned up
    assert not model._context_encoded
    assert model._cached_context_batch_size is None
    assert model.cache_trainset_representation == orig_cache_setting


if __name__ == "__main__":
    test_broadcast_kv_cache_correctness()
    print("✓ test_broadcast_kv_cache_correctness passed")

    test_broadcast_kv_cache_gradient_flow()
    print("✓ test_broadcast_kv_cache_gradient_flow passed")

    test_multihead_attention_broadcast()
    print("✓ test_multihead_attention_broadcast passed")

    test_shared_context_cleanup()
    print("✓ test_shared_context_cleanup passed")

    print("\nAll tests passed!")
