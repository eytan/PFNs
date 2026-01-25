# PFNModel Caching Refactor Plan

## Overview

This document provides a comprehensive plan for refactoring `PFNModel` and its subclasses to properly support KV caching. The goal is to enable memory-efficient batch evaluation during BoTorch acquisition function optimization.

### Current Issues

1. **Implicit state**: `_training_cache_ready` and `_cached_negate_train_ys` are not initialized in `__init__`
2. **No capability check**: No validation that the transformer supports caching
3. **Subclass incompatibility**: `PFNModelWithPendingPoints` and `MultivariatePFNModel` don't properly support caching
4. **Shape mismatch**: `_prepare_data` returns wrong-shaped tensors when `skip_train_replication=True`
5. **Benchmark bugs**: Gradient verification uses inconsistent test shapes

### Design Decisions

1. **Template Method Pattern**: Refactor `posterior()` into composable hooks
2. **Pending Points Caching**: Pending points are training data with NaN labels; cache is valid if pending locations don't change
3. **Cache Validation**: Store hash of cached training context (including pending) for validation

---

## Part 0: Correctness Verification (MUST DO FIRST)

**Rationale**: The current implementation is hacky and the existing benchmark "tests" have flaws that could mask bugs. Before any refactoring, we must verify the current implementation is actually correct.

### Known Benchmark Flaws

1. **Loose tolerances**: Benchmarks use 0.01 tolerance, masking potential precision issues
2. **Wrong test shape in gradient verification**: Cache populated with shape `[1, 1, d]` but tested with `[1, n_test, d]`
3. **No verification cache is engaged**: Benchmarks don't assert caching is active
4. **Optimization stochasticity conflated with correctness**: `optimize_acqf` results compared but differences could be optimization noise OR bugs

### 0.1 Create Rigorous Correctness Tests

Create `tests/test_current_caching_correctness.py` - tests that will FAIL if caching is broken:

```python
"""Rigorous correctness tests for current PFNModel caching implementation.

These tests verify the CURRENT implementation works before refactoring.
They use strict tolerances and proper test design to catch real bugs.
"""

import pytest
import torch

from pfns.botorch_pfn import PFNModel
from pfns.model.bar_distribution import BarDistribution
from pfns.model.transformer import TableTransformer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    return torch.float32


@pytest.fixture
def transformer(device, dtype):
    """Create a deterministic transformer for testing."""
    torch.manual_seed(42)
    num_buckets = 50
    borders = torch.linspace(-3, 3, num_buckets + 1)

    model = TableTransformer(
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=4,
        batch_first=True,
        cache_trainset_representation=False,
        decoder_dict={"standard": (None, num_buckets)},
    )
    model.criterion = BarDistribution(borders=borders)
    return model.to(device=device, dtype=dtype).eval()


@pytest.fixture
def training_data(device, dtype):
    """Create deterministic training data."""
    torch.manual_seed(123)
    n_train, n_dims = 30, 5
    train_X = torch.rand(n_train, n_dims, device=device, dtype=dtype)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) + 0.1 * torch.randn(
        n_train, 1, device=device, dtype=dtype
    )
    train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-8)
    return train_X, train_Y


class TestCachingNumericalEquivalence:
    """Test that cached and non-cached produce IDENTICAL results."""

    STRICT_TOLERANCE = 1e-6  # Much stricter than benchmark's 0.01

    def test_single_point_prediction_identical(
        self, transformer, training_data, device, dtype
    ):
        """Single test point: cached == non-cached."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        torch.manual_seed(456)
        test_X = torch.rand(1, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached
        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            probs_std = posterior_std.probabilities.clone()
            mean_std = posterior_std.mean.clone()
            var_std = posterior_std.variance.clone()

        # Cached
        with model.cache_training_context():
            # VERIFY CACHE IS ACTUALLY ENABLED
            assert model._training_cache_ready, "Cache not enabled!"
            assert model.pfn.cache_trainset_representation, "Transformer cache not enabled!"

            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                probs_cached = posterior_cached.probabilities.clone()
                mean_cached = posterior_cached.mean.clone()
                var_cached = posterior_cached.variance.clone()

        # STRICT comparison
        prob_diff = (probs_std - probs_cached).abs().max().item()
        mean_diff = (mean_std - mean_cached).abs().max().item()
        var_diff = (var_std - var_cached).abs().max().item()

        assert prob_diff < self.STRICT_TOLERANCE, (
            f"Probabilities differ: max diff = {prob_diff}"
        )
        assert mean_diff < self.STRICT_TOLERANCE, (
            f"Means differ: max diff = {mean_diff}"
        )
        assert var_diff < self.STRICT_TOLERANCE, (
            f"Variances differ: max diff = {var_diff}"
        )

    def test_batch_prediction_identical(
        self, transformer, training_data, device, dtype
    ):
        """Batched test points: cached == non-cached."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        torch.manual_seed(789)
        # Shape that mimics optimize_acqf: (num_restarts, q, d)
        test_X = torch.rand(16, 1, train_X.shape[-1], device=device, dtype=dtype)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()

        with model.cache_training_context():
            assert model._training_cache_ready
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()

        mean_diff = (mean_std - mean_cached).abs().max().item()
        assert mean_diff < self.STRICT_TOLERANCE, (
            f"Batched means differ: max diff = {mean_diff}"
        )

    def test_multiple_calls_within_context_identical(
        self, transformer, training_data, device, dtype
    ):
        """Multiple posterior() calls in same context should all match."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        torch.manual_seed(111)
        test_X1 = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)
        test_X2 = torch.rand(8, train_X.shape[-1], device=device, dtype=dtype)
        test_X3 = torch.rand(3, 2, train_X.shape[-1], device=device, dtype=dtype)

        # Get non-cached results
        with torch.no_grad():
            mean_std1 = model.posterior(test_X1).mean.clone()
            mean_std2 = model.posterior(test_X2).mean.clone()
            mean_std3 = model.posterior(test_X3).mean.clone()

        # Get cached results
        with model.cache_training_context():
            with torch.no_grad():
                mean_cached1 = model.posterior(test_X1).mean.clone()
                mean_cached2 = model.posterior(test_X2).mean.clone()
                mean_cached3 = model.posterior(test_X3).mean.clone()

        for i, (std, cached) in enumerate([
            (mean_std1, mean_cached1),
            (mean_std2, mean_cached2),
            (mean_std3, mean_cached3),
        ], 1):
            diff = (std - cached).abs().max().item()
            assert diff < self.STRICT_TOLERANCE, (
                f"Call {i} differs: max diff = {diff}"
            )


class TestCachingGradientCorrectness:
    """Test gradient flow is correct with caching."""

    GRAD_TOLERANCE = 1e-5

    def test_gradient_values_match(
        self, transformer, training_data, device, dtype
    ):
        """Gradients through cached path should match non-cached."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        torch.manual_seed(222)
        base_test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached gradient
        test_X_std = base_test_X.clone().requires_grad_(True)
        posterior_std = model.posterior(test_X_std)
        loss_std = posterior_std.mean.sum()
        loss_std.backward()
        grad_std = test_X_std.grad.clone()

        # Cached gradient
        test_X_cached = base_test_X.clone().requires_grad_(True)
        with model.cache_training_context():
            assert model._training_cache_ready
            posterior_cached = model.posterior(test_X_cached)
            loss_cached = posterior_cached.mean.sum()
            loss_cached.backward()
        grad_cached = test_X_cached.grad.clone()

        # Verify gradients exist and are non-zero
        assert grad_std is not None, "Non-cached gradient is None"
        assert grad_cached is not None, "Cached gradient is None"
        assert not torch.all(grad_std == 0), "Non-cached gradient is all zeros"
        assert not torch.all(grad_cached == 0), "Cached gradient is all zeros"

        # Verify gradients match
        grad_diff = (grad_std - grad_cached).abs().max().item()
        assert grad_diff < self.GRAD_TOLERANCE, (
            f"Gradients differ: max diff = {grad_diff}"
        )

    def test_gradient_shape_consistency(
        self, transformer, training_data, device, dtype
    ):
        """Gradients should have correct shape for various input shapes."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        test_shapes = [
            (5, train_X.shape[-1]),        # (q, d)
            (8, 1, train_X.shape[-1]),     # (b, q, d) with q=1
            (4, 3, train_X.shape[-1]),     # (b, q, d) with q>1
        ]

        for shape in test_shapes:
            test_X = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)

            with model.cache_training_context():
                posterior = model.posterior(test_X)
                posterior.mean.sum().backward()

            assert test_X.grad is not None, f"No gradient for shape {shape}"
            assert test_X.grad.shape == shape, (
                f"Wrong gradient shape for input {shape}: got {test_X.grad.shape}"
            )


class TestCachingEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_cache_state_cleanup_on_normal_exit(
        self, transformer, training_data, device, dtype
    ):
        """Cache state should be fully cleaned up after context exits."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        with model.cache_training_context():
            assert model._training_cache_ready

        # Verify cleanup
        assert not model._training_cache_ready
        assert not model.pfn.cache_trainset_representation

    def test_cache_state_cleanup_on_exception(
        self, transformer, training_data, device, dtype
    ):
        """Cache state should be cleaned up even if exception occurs."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            with model.cache_training_context():
                assert model._training_cache_ready
                raise TestException("Intentional")

        # Verify cleanup happened despite exception
        assert not model._training_cache_ready

    def test_nested_cache_context_behavior(
        self, transformer, training_data, device, dtype
    ):
        """Nested cache contexts should be handled safely."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        # This tests what happens with nested contexts
        # Current implementation may or may not support this
        with model.cache_training_context():
            assert model._training_cache_ready

            # Nested context - behavior should be defined
            # Either: work correctly, or raise clear error
            try:
                with model.cache_training_context():
                    pass
            except Exception as e:
                # If it raises, that's acceptable - document the behavior
                print(f"Nested context raises: {type(e).__name__}: {e}")

        assert not model._training_cache_ready

    def test_negate_train_ys_cached_correctly(
        self, transformer, training_data, device, dtype
    ):
        """Negation setting should be respected in cache."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        torch.manual_seed(333)
        test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached with negation
        with torch.no_grad():
            mean_std = model.posterior(test_X, negate_train_ys=True).mean.clone()

        # Cached with negation
        with model.cache_training_context(negate_train_ys=True):
            with torch.no_grad():
                mean_cached = model.posterior(test_X, negate_train_ys=True).mean.clone()

        diff = (mean_std - mean_cached).abs().max().item()
        assert diff < 1e-6, f"Negated predictions differ: {diff}"

    def test_different_batch_sizes_same_cache(
        self, transformer, training_data, device, dtype
    ):
        """Different batch sizes should work with same cache."""
        train_X, train_Y = training_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=transformer,
            batch_first=True,
        )

        with model.cache_training_context():
            # Various batch sizes
            for batch_size in [1, 4, 16, 32]:
                test_X = torch.rand(
                    batch_size, 1, train_X.shape[-1], device=device, dtype=dtype
                )
                with torch.no_grad():
                    posterior = model.posterior(test_X)
                    # Should not raise
                    _ = posterior.mean


class TestTransformerLevelCaching:
    """Verify the underlying transformer caching works correctly."""

    STRICT_TOLERANCE = 1e-6

    def test_transformer_cache_direct(self, transformer, training_data, device, dtype):
        """Test transformer caching directly, bypassing PFNModel."""
        train_X, train_Y = training_data

        # Prepare data in transformer format
        train_x = train_X.unsqueeze(0)  # (1, n, d)
        train_y = train_Y.unsqueeze(0)  # (1, n, 1)

        torch.manual_seed(444)
        test_x = torch.rand(8, 3, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached
        transformer.cache_trainset_representation = False
        with torch.no_grad():
            output_std = transformer(
                x=train_x.expand(8, -1, -1),
                y=train_y.expand(8, -1, -1),
                test_x=test_x,
            ).clone()

        # Cached
        transformer.cache_trainset_representation = True
        transformer.empty_trainset_representation_cache()

        # Populate cache
        dummy_test = torch.zeros(1, 1, train_X.shape[-1], device=device, dtype=dtype)
        with torch.no_grad():
            _ = transformer(x=train_x, y=train_y, test_x=dummy_test)

        # Use cache
        with torch.no_grad():
            output_cached = transformer(x=None, y=None, test_x=test_x).clone()

        transformer.empty_trainset_representation_cache()
        transformer.cache_trainset_representation = False

        diff = (output_std - output_cached).abs().max().item()
        assert diff < self.STRICT_TOLERANCE, (
            f"Transformer outputs differ: {diff}"
        )
```

### 0.2 Run Correctness Tests

```bash
# Run the correctness tests
pytest tests/test_current_caching_correctness.py -v --tb=short

# If any fail, the current implementation has bugs that must be fixed FIRST
```

### 0.3 Decision Point

**If all tests pass**: Proceed to Part 1 (refactoring)

**If tests fail**:
1. Document which tests fail and how
2. Fix the bugs in the current implementation
3. Re-run tests until all pass
4. Then proceed to Part 1

### 0.4 Known Issues to Investigate

Based on code review, these might cause test failures:

1. **`_prepare_data` shape mismatch**: Returns `(n, d)` shaped tensors when `skip_train_replication=True` but comments say "won't be used" - if they ARE used, this would cause silent bugs

2. **Batch broadcasting in cache**: Cache is populated with batch_size=1, but inference uses batch_size=b. Does the transformer correctly broadcast the cached KV?

3. **Styles with caching**: `_get_styles(batch_size=1)` during cache population, but `_get_styles(batch_size=b)` during inference. Are styles correctly broadcast?

4. **`batch_first` transpose logic**: Duplicated in cached and non-cached paths - easy to have inconsistency

---

## Part 1: Template Method Refactor

### Goal

Refactor `PFNModel.posterior()` so subclasses override specific hooks rather than the entire method.

### File: `pfns/botorch_pfn.py`

#### 1.1 Add Cache State Initialization

In `PFNModel.__init__`, after line 187 (after `self._compute_styles()`), add:

```python
# Cache state initialization
self._training_cache_ready = False
self._cached_negate_train_ys = False
self._cached_context_hash: int | None = None
```

#### 1.2 Add Caching Capability Check

In `cache_training_context()`, before line 233 (before `self.pfn.cache_trainset_representation = True`), add:

```python
# Validate transformer supports caching
if not hasattr(self.pfn, "cache_trainset_representation"):
    raise TypeError(
        f"Model {type(self.pfn).__name__} does not support caching. "
        "Requires 'cache_trainset_representation' attribute."
    )
if not callable(getattr(self.pfn, "empty_trainset_representation_cache", None)):
    raise TypeError(
        f"Model {type(self.pfn).__name__} does not support caching. "
        "Requires 'empty_trainset_representation_cache()' method."
    )
```

#### 1.3 Add Context Hashing Helper

Add new method after `_compute_styles()`:

```python
def _compute_context_hash(
    self,
    train_X: Tensor,
    train_Y: Tensor,
    negate_train_ys: bool
) -> int:
    """Compute hash of training context for cache validation."""
    # Use shape and a sample of values for fast hashing
    shape_tuple = (train_X.shape, train_Y.shape, negate_train_ys)
    # Sample corners and center for value check (avoid hashing entire tensor)
    x_sample = (
        train_X[0, 0].item() if train_X.numel() > 0 else 0,
        train_X[-1, -1].item() if train_X.numel() > 0 else 0,
    )
    y_sample = (
        train_Y[0, 0].item() if train_Y.numel() > 0 else 0,
        train_Y[-1, 0].item() if train_Y.numel() > 0 else 0,
    )
    return hash((shape_tuple, x_sample, y_sample))
```

#### 1.4 Refactor `posterior()` to Template Method

Replace the current `posterior()` method (lines 288-358) with:

```python
def posterior(
    self,
    X: Tensor,
    output_indices: Optional[list[int]] = None,
    observation_noise: Union[bool, Tensor] = False,
    posterior_transform: Optional[PosteriorTransform] = None,
    negate_train_ys: bool = False,
) -> BoundedRiemannPosterior:
    r"""Computes the posterior over model outputs at the provided points.

    Subclasses should override hooks (_augment_training_data, _build_posterior)
    rather than this method directly.

    Args:
        X: A ``b? x q? x d``-dim Tensor.
        output_indices: **Not supported.**
        observation_noise: **Not supported.**
        posterior_transform: **Not supported.**
        negate_train_ys: Whether to negate training Ys (for minimization).

    Returns:
        A ``BoundedRiemannPosterior``.
    """
    self.pfn.eval()
    self._validate_posterior_args(output_indices, observation_noise, posterior_transform)

    # Check cache state
    use_cache = self._check_cache_compatibility(negate_train_ys)

    # Prepare base training data
    X, train_X, train_Y, orig_X_shape, styles = self._prepare_data(
        X, negate_train_ys=negate_train_ys, skip_train_replication=use_cache
    )

    # Hook: Allow subclasses to augment training data (e.g., pending points)
    train_X, train_Y = self._augment_training_data(
        train_X, train_Y, X, negate_train_ys=negate_train_ys, use_cache=use_cache
    )

    # Validate cache matches current context (including any augmentation)
    if use_cache:
        self._validate_cache_context(train_X, train_Y, negate_train_ys)

    # Core prediction
    probabilities = self.pfn_predict(
        X=X,
        train_X=train_X,
        train_Y=train_Y,
        use_cache=use_cache,
        **self.constant_model_kwargs,
        **styles,
    )
    probabilities = probabilities.view(*orig_X_shape[:-1], -1)

    # Hook: Allow subclasses to build custom posteriors
    return self._build_posterior(
        X=X,
        probabilities=probabilities,
        orig_X_shape=orig_X_shape,
        train_X=train_X,
        train_Y=train_Y,
        styles=styles,
        use_cache=use_cache,
    )
```

#### 1.5 Add Validation Helper

Add after `posterior()`:

```python
def _validate_posterior_args(
    self,
    output_indices: Optional[list[int]],
    observation_noise: Union[bool, Tensor],
    posterior_transform: Optional[PosteriorTransform],
) -> None:
    """Validate unsupported posterior arguments."""
    if output_indices is not None:
        raise UnsupportedError(
            "output_indices is not None. PFNModel should not be a multi-output model."
        )
    if observation_noise:
        logger.warning(
            "observation_noise is not supported for PFNModel and is being ignored."
        )
    if posterior_transform is not None:
        raise UnsupportedError("posterior_transform is not supported for PFNModel.")
```

#### 1.6 Add Cache Compatibility Check

```python
def _check_cache_compatibility(self, negate_train_ys: bool) -> bool:
    """Check if cache can be used and validate settings."""
    if not self._training_cache_ready:
        return False

    if negate_train_ys != self._cached_negate_train_ys:
        raise ValueError(
            f"negate_train_ys={negate_train_ys} does not match cached "
            f"negate_train_ys={self._cached_negate_train_ys}. "
            "The cache was populated with a different negation setting."
        )
    return True
```

#### 1.7 Add Cache Context Validation

```python
def _validate_cache_context(
    self, train_X: Tensor, train_Y: Tensor, negate_train_ys: bool
) -> None:
    """Validate that current context matches cached context."""
    current_hash = self._compute_context_hash(train_X, train_Y, negate_train_ys)
    if current_hash != self._cached_context_hash:
        raise ValueError(
            "Training context has changed since cache was populated. "
            "This can happen if pending points changed. "
            "Exit and re-enter cache_training_context() to refresh the cache."
        )
```

#### 1.8 Add Template Method Hooks

```python
def _augment_training_data(
    self,
    train_X: Tensor,
    train_Y: Tensor,
    X: Tensor,
    *,
    negate_train_ys: bool,
    use_cache: bool,
) -> tuple[Tensor, Tensor]:
    """Hook for subclasses to augment training data.

    Override this instead of posterior() to add pending points, etc.

    Args:
        train_X: Training features, shape (b, n, d) or (n, d) if use_cache
        train_Y: Training targets, shape (b, n, 1) or (n, 1) if use_cache
        X: Test points, shape (b, q, d)
        negate_train_ys: Whether train_Y was negated
        use_cache: Whether caching is active

    Returns:
        Potentially augmented (train_X, train_Y)
    """
    return train_X, train_Y

def _build_posterior(
    self,
    X: Tensor,
    probabilities: Tensor,
    orig_X_shape: torch.Size,
    train_X: Tensor,
    train_Y: Tensor,
    styles: dict[str, Tensor],
    use_cache: bool,
) -> BoundedRiemannPosterior:
    """Hook for subclasses to build custom posteriors.

    Override this instead of posterior() for multivariate posteriors, etc.
    """
    return BoundedRiemannPosterior(
        borders=self.borders,
        probabilities=probabilities,
    )
```

#### 1.9 Update `_prepare_data` Signature

Update `_prepare_data` to return `None` instead of wrong-shaped placeholders:

```python
def _prepare_data(
    self,
    X: Tensor,
    negate_train_ys: bool = False,
    skip_train_replication: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, torch.Size, dict[str, Tensor]]:
    """Prepare data for posterior computation.

    Returns:
        X: Transformed test points (b, q, d)
        train_X: Training features (b, n, d) or None if skip_train_replication
        train_Y: Training targets (b, n, 1) or None if skip_train_replication
        orig_X_shape: Original shape of X before unsqueezing
        styles: Style tensors for the model
    """
    orig_X_shape = X.shape
    if len(X.shape) > 3:
        raise UnsupportedError(f"X must be at most 3-d, got {X.shape}.")
    while len(X.shape) < 3:
        X = X.unsqueeze(0)

    X = self.transform_inputs(X)

    if skip_train_replication:
        train_X = None
        train_Y = None
    else:
        train_X = match_batch_shape(self.transformed_X, X)
        train_Y = match_batch_shape(self.train_Y, X)
        if negate_train_ys:
            assert self.train_Y.mean().abs() < 1e-4, "train_Y must be zero-centered."
            train_Y = -train_Y

    styles = self._get_styles(batch_size=X.shape[0])
    return X, train_X, train_Y, orig_X_shape, styles
```

#### 1.10 Update `pfn_predict` Signature

Add explicit `use_cache` parameter:

```python
def pfn_predict(
    self,
    X: Tensor,
    train_X: Tensor | None,
    train_Y: Tensor | None,
    use_cache: bool = False,
    **forward_kwargs,
) -> Tensor:
    """Make a prediction using the PFN model.

    Args:
        X: Test points, shape (b, q, d)
        train_X: Training features, shape (b, n, d) or None if use_cache
        train_Y: Training targets, shape (b, n, 1) or None if use_cache
        use_cache: Whether to use cached training representations
        **forward_kwargs: Additional kwargs for the PFN model

    Returns:
        Probabilities (b, q, num_buckets)
    """
    if use_cache:
        assert train_X is None and train_Y is None, (
            "Bug: use_cache=True but train_X/train_Y provided"
        )
        if not self.batch_first:
            X = X.transpose(0, 1)

        logits = self.pfn(
            x=None,
            y=None,
            test_x=X.float(),
            **forward_kwargs,
        )

        if not self.batch_first:
            logits = logits.transpose(0, 1)
    else:
        assert train_X is not None and train_Y is not None, (
            "Bug: use_cache=False but train_X/train_Y is None"
        )
        if not self.batch_first:
            X = X.transpose(0, 1)
            train_X = train_X.transpose(0, 1)
            train_Y = train_Y.transpose(0, 1)

        logits = self.pfn(
            x=train_X.float(),
            y=train_Y.float(),
            test_x=X.float(),
            **forward_kwargs,
        )

        if not self.batch_first:
            logits = logits.transpose(0, 1)

    logits = logits.to(X.dtype)
    probabilities = logits.softmax(dim=-1)
    return probabilities
```

#### 1.11 Update `cache_training_context`

Modify to support pending points and compute context hash:

```python
@contextmanager
def cache_training_context(
    self,
    negate_train_ys: bool = False,
    pending_X: Tensor | None = None,
) -> Iterator[None]:
    """Context manager to cache training data for efficient batch evaluation.

    When used with BoTorch's ``optimize_acqf()``, this caches the training
    data representations once, allowing many test point evaluations without
    recomputing the training context.

    Args:
        negate_train_ys: Whether to negate training Y values (for minimization).
        pending_X: Optional pending points to include in the cache. Shape (n', d).
            If provided, all posterior() calls within this context must use
            the same pending points.
    """
    # Validate transformer supports caching
    if not hasattr(self.pfn, "cache_trainset_representation"):
        raise TypeError(
            f"Model {type(self.pfn).__name__} does not support caching."
        )
    if not callable(getattr(self.pfn, "empty_trainset_representation_cache", None)):
        raise TypeError(
            f"Model {type(self.pfn).__name__} does not support caching."
        )

    original_cache_state = getattr(self.pfn, "cache_trainset_representation", False)

    try:
        self.pfn.cache_trainset_representation = True

        # Prepare training data (single batch)
        train_X = self.transformed_X.unsqueeze(0)  # (1, n, d)
        train_Y = self.train_Y.unsqueeze(0)  # (1, n, 1)

        if negate_train_ys:
            assert self.train_Y.mean().abs() < 1e-4, (
                "train_Y must be zero-centered for negation."
            )
            train_Y = -train_Y

        # Add pending points if provided
        if pending_X is not None:
            if pending_X.dim() != 2:
                raise ValueError("pending_X must be 2-dimensional (n', d)")
            pending_X_batch = pending_X.unsqueeze(0)  # (1, n', d)
            train_X = torch.cat([train_X, pending_X_batch], dim=1)
            train_Y = torch.cat([
                train_Y,
                torch.full(
                    (1, pending_X.shape[0], 1),
                    torch.nan,
                    device=train_Y.device,
                    dtype=train_Y.dtype,
                ),
            ], dim=1)

        # Compute context hash for validation
        self._cached_context_hash = self._compute_context_hash(
            train_X.squeeze(0), train_Y.squeeze(0), negate_train_ys
        )

        # Get styles
        styles = self._get_styles(batch_size=1)

        # Transpose if needed
        if not self.batch_first:
            train_X = train_X.transpose(0, 1)
            train_Y = train_Y.transpose(0, 1)
            dummy_test = torch.zeros(1, 1, train_X.shape[-1], device=train_X.device, dtype=train_X.dtype)
        else:
            dummy_test = torch.zeros(1, 1, train_X.shape[-1], device=train_X.device, dtype=train_X.dtype)

        # Populate cache
        with torch.no_grad():
            self.pfn(
                x=train_X.float(),
                y=train_Y.float(),
                test_x=dummy_test.float(),
                **self.constant_model_kwargs,
                **styles,
            )

        self._training_cache_ready = True
        self._cached_negate_train_ys = negate_train_ys
        self._cached_pending_X = pending_X  # Store for validation

        yield

    finally:
        self._training_cache_ready = False
        self._cached_negate_train_ys = False
        self._cached_context_hash = None
        self._cached_pending_X = None
        self.pfn.cache_trainset_representation = original_cache_state
        if hasattr(self.pfn, "empty_trainset_representation_cache"):
            self.pfn.empty_trainset_representation_cache()
```

---

## Part 2: Subclass Refactoring

### 2.1 Refactor `PFNModelWithPendingPoints`

Replace the entire class with:

```python
class PFNModelWithPendingPoints(PFNModel):
    """PFNModel that supports pending points (unobserved evaluations).

    Pending points are added to the training context with NaN labels,
    allowing the model to account for in-flight evaluations.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        pending_X: Tensor | None = None,
        **kwargs,
    ) -> None:
        """Initialize with optional pending points.

        Args:
            train_X: Training features (n, d)
            train_Y: Training targets (n, 1)
            pending_X: Optional pending point locations (n', d)
            **kwargs: Additional arguments for PFNModel
        """
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._pending_X = pending_X

    @property
    def pending_X(self) -> Tensor | None:
        """Current pending points."""
        return self._pending_X

    @pending_X.setter
    def pending_X(self, value: Tensor | None) -> None:
        """Set pending points. Invalidates any active cache."""
        if self._training_cache_ready:
            raise RuntimeError(
                "Cannot modify pending_X while cache_training_context is active. "
                "Exit the context manager first."
            )
        self._pending_X = value

    def _augment_training_data(
        self,
        train_X: Tensor | None,
        train_Y: Tensor | None,
        X: Tensor,
        *,
        negate_train_ys: bool,
        use_cache: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Add pending points to training data."""
        if self._pending_X is None:
            return train_X, train_Y

        if use_cache:
            # When using cache, pending points were included at cache creation
            # Just return None to signal cached path
            return None, None

        # Non-cached path: augment training data with pending points
        assert train_X is not None and train_Y is not None

        pending_X = self._pending_X[None].expand(X.shape[0], -1, -1)  # (b, n', d)
        train_X = torch.cat([train_X, pending_X], dim=1)  # (b, n+n', d)
        train_Y = torch.cat([
            train_Y,
            torch.full(
                (train_Y.shape[0], pending_X.shape[1], 1),
                torch.nan,
                device=train_Y.device,
                dtype=train_Y.dtype,
            ),
        ], dim=1)  # (b, n+n', 1)

        return train_X, train_Y

    def cache_training_context(
        self,
        negate_train_ys: bool = False,
        pending_X: Tensor | None = None,
    ) -> Iterator[None]:
        """Cache training context including pending points.

        Args:
            negate_train_ys: Whether to negate training Y values.
            pending_X: Override pending points for this cache context.
                If None, uses self.pending_X.
        """
        # Use instance pending_X if not overridden
        effective_pending = pending_X if pending_X is not None else self._pending_X
        return super().cache_training_context(
            negate_train_ys=negate_train_ys,
            pending_X=effective_pending,
        )
```

### 2.2 Refactor `MultivariatePFNModel`

Replace the `posterior` method override with `_build_posterior` hook:

```python
class MultivariatePFNModel(PFNModel):
    """PFN model returning joint posterior over q batch inputs."""

    def _build_posterior(
        self,
        X: Tensor,
        probabilities: Tensor,
        orig_X_shape: torch.Size,
        train_X: Tensor | None,
        train_Y: Tensor | None,
        styles: dict[str, Tensor],
        use_cache: bool,
    ) -> BoundedRiemannPosterior | MultivariateRiemannPosterior:
        """Build multivariate posterior with correlation estimation."""
        marginals = BoundedRiemannPosterior(
            borders=self.borders,
            probabilities=probabilities,
        )

        # If no q dimension or q=1, return marginals
        if len(orig_X_shape) == 1 or orig_X_shape[-2] == 1:
            return marginals

        # For correlation estimation, we need full training data
        if train_X is None or train_Y is None:
            # Re-fetch training data without cache for correlation estimation
            _, train_X, train_Y, _, _ = self._prepare_data(
                X.view(*orig_X_shape),
                skip_train_replication=False
            )
            # Apply any augmentation (e.g., pending points)
            train_X, train_Y = self._augment_training_data(
                train_X, train_Y, X,
                negate_train_ys=self._cached_negate_train_ys if use_cache else False,
                use_cache=False,  # Force non-cached for correlation
            )

        R = self.estimate_correlations(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            styles=styles,
            marginals=marginals,
        )
        R = R.view(*orig_X_shape[:-2], X.shape[-2], X.shape[-2])

        return MultivariateRiemannPosterior(
            borders=self.borders,
            probabilities=marginals.probabilities,
            correlation_matrix=R,
        )

    # Keep existing estimate_correlations, _compute_conditional_means,
    # _estimate_covariances, _map_psd methods unchanged
```

---

## Part 3: Test Plan

### File: `tests/test_botorch_pfn_caching.py`

```python
"""Tests for PFNModel caching functionality."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from pfns.botorch_pfn import (
    PFNModel,
    PFNModelWithPendingPoints,
    MultivariatePFNModel,
)
from pfns.model.bar_distribution import BarDistribution
from pfns.model.transformer import TableTransformer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def simple_transformer(device):
    """Create a simple transformer for testing."""
    num_buckets = 10
    borders = torch.linspace(-3, 3, num_buckets + 1)

    model = TableTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        batch_first=True,
        cache_trainset_representation=False,
        decoder_dict={"standard": (None, num_buckets)},
    )
    model.criterion = BarDistribution(borders=borders)
    return model.to(device).eval()


@pytest.fixture
def train_data(device):
    """Create training data for tests."""
    torch.manual_seed(42)
    n_train, n_dims = 20, 5
    train_X = torch.rand(n_train, n_dims, device=device)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()
    return train_X, train_Y


class TestPFNModelCacheInitialization:
    """Test cache state initialization."""

    def test_cache_state_initialized_in_init(self, simple_transformer, train_data):
        """Cache state should be explicitly initialized."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        assert hasattr(model, "_training_cache_ready")
        assert model._training_cache_ready is False
        assert hasattr(model, "_cached_negate_train_ys")
        assert model._cached_negate_train_ys is False
        assert hasattr(model, "_cached_context_hash")
        assert model._cached_context_hash is None


class TestPFNModelCacheCapabilityCheck:
    """Test validation of transformer caching capability."""

    def test_raises_on_missing_cache_attribute(self, train_data):
        """Should raise if transformer doesn't support caching."""
        train_X, train_Y = train_data

        # Create mock model without caching support
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        del mock_model.cache_trainset_representation  # Ensure missing
        mock_model.criterion = MagicMock()
        mock_model.criterion.borders = torch.linspace(-3, 3, 11)

        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=mock_model,
            batch_first=True,
        )

        with pytest.raises(TypeError, match="does not support caching"):
            with model.cache_training_context():
                pass

    def test_raises_on_missing_cache_clear_method(self, train_data):
        """Should raise if transformer lacks cache clearing method."""
        train_X, train_Y = train_data

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.cache_trainset_representation = False
        del mock_model.empty_trainset_representation_cache
        mock_model.criterion = MagicMock()
        mock_model.criterion.borders = torch.linspace(-3, 3, 11)

        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=mock_model,
            batch_first=True,
        )

        with pytest.raises(TypeError, match="does not support caching"):
            with model.cache_training_context():
                pass


class TestPFNModelCacheContextManager:
    """Test cache_training_context behavior."""

    def test_cache_enabled_within_context(self, simple_transformer, train_data, device):
        """Cache should be enabled within context."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        assert not model._training_cache_ready

        with model.cache_training_context():
            assert model._training_cache_ready
            assert model.pfn.cache_trainset_representation

        assert not model._training_cache_ready

    def test_cache_cleaned_on_exit(self, simple_transformer, train_data, device):
        """Cache should be cleared when exiting context."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        with model.cache_training_context():
            pass

        assert not model._training_cache_ready
        assert model._cached_context_hash is None

    def test_cache_cleaned_on_exception(self, simple_transformer, train_data, device):
        """Cache should be cleared even if exception occurs."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        with pytest.raises(RuntimeError):
            with model.cache_training_context():
                assert model._training_cache_ready
                raise RuntimeError("Test exception")

        assert not model._training_cache_ready


class TestPFNModelCachedPredictions:
    """Test that cached predictions match non-cached."""

    def test_predictions_match(self, simple_transformer, train_data, device):
        """Cached and non-cached predictions should be identical."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_X = torch.rand(5, train_X.shape[-1], device=device)

        # Non-cached
        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean
            var_std = posterior_std.variance

        # Cached
        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean
                var_cached = posterior_cached.variance

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(var_std, var_cached, atol=1e-5, rtol=1e-5)

    def test_predictions_match_batched(self, simple_transformer, train_data, device):
        """Cached predictions should work with batched inputs."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_X = torch.rand(8, 3, train_X.shape[-1], device=device)  # (b, q, d)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)

        torch.testing.assert_close(
            posterior_std.mean, posterior_cached.mean, atol=1e-5, rtol=1e-5
        )


class TestPFNModelCacheGradients:
    """Test gradient flow through cached predictions."""

    def test_gradients_flow_through_test_x(self, simple_transformer, train_data, device):
        """Gradients should flow through test_x when caching."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_X = torch.rand(5, train_X.shape[-1], device=device, requires_grad=True)

        with model.cache_training_context():
            posterior = model.posterior(test_X)
            loss = posterior.mean.sum()
            loss.backward()

        assert test_X.grad is not None
        assert not torch.all(test_X.grad == 0)

    def test_gradients_match_non_cached(self, simple_transformer, train_data, device):
        """Gradients should match between cached and non-cached."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        # Non-cached gradient
        test_X_std = torch.rand(5, train_X.shape[-1], device=device, requires_grad=True)
        posterior_std = model.posterior(test_X_std)
        posterior_std.mean.sum().backward()
        grad_std = test_X_std.grad.clone()

        # Cached gradient
        test_X_cached = test_X_std.detach().clone().requires_grad_(True)
        with model.cache_training_context():
            posterior_cached = model.posterior(test_X_cached)
            posterior_cached.mean.sum().backward()
        grad_cached = test_X_cached.grad.clone()

        torch.testing.assert_close(grad_std, grad_cached, atol=1e-4, rtol=1e-4)


class TestPFNModelCacheNegation:
    """Test cache behavior with negate_train_ys."""

    def test_negate_mismatch_raises(self, simple_transformer, train_data, device):
        """Should raise if negate_train_ys doesn't match cache."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_X = torch.rand(5, train_X.shape[-1], device=device)

        with model.cache_training_context(negate_train_ys=False):
            with pytest.raises(ValueError, match="negate_train_ys"):
                model.posterior(test_X, negate_train_ys=True)


class TestPFNModelWithPendingPointsCaching:
    """Test caching with pending points."""

    def test_cache_includes_pending_points(self, simple_transformer, train_data, device):
        """Cache should include pending points."""
        train_X, train_Y = train_data
        pending_X = torch.rand(3, train_X.shape[-1], device=device)

        model = PFNModelWithPendingPoints(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            pending_X=pending_X,
            batch_first=True,
        )

        test_X = torch.rand(5, train_X.shape[-1], device=device)

        # Non-cached
        with torch.no_grad():
            posterior_std = model.posterior(test_X)

        # Cached (should include pending points)
        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)

        torch.testing.assert_close(
            posterior_std.mean, posterior_cached.mean, atol=1e-5, rtol=1e-5
        )

    def test_pending_modification_blocked_during_cache(
        self, simple_transformer, train_data, device
    ):
        """Cannot modify pending_X while cache is active."""
        train_X, train_Y = train_data
        pending_X = torch.rand(3, train_X.shape[-1], device=device)

        model = PFNModelWithPendingPoints(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            pending_X=pending_X,
            batch_first=True,
        )

        with model.cache_training_context():
            with pytest.raises(RuntimeError, match="Cannot modify pending_X"):
                model.pending_X = torch.rand(2, train_X.shape[-1], device=device)


class TestMultivariatePFNModelCaching:
    """Test caching with multivariate posteriors."""

    def test_marginals_use_cache(self, simple_transformer, train_data, device):
        """Marginal computation should use cache."""
        train_X, train_Y = train_data
        model = MultivariatePFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        # Single point - should return BoundedRiemannPosterior
        test_X = torch.rand(1, train_X.shape[-1], device=device)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)

        torch.testing.assert_close(
            posterior_std.mean, posterior_cached.mean, atol=1e-5, rtol=1e-5
        )

    def test_multivariate_with_cache(self, simple_transformer, train_data, device):
        """Multivariate posterior should work with caching."""
        train_X, train_Y = train_data
        model = MultivariatePFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        # Multiple points - should return MultivariateRiemannPosterior
        test_X = torch.rand(3, train_X.shape[-1], device=device)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)

        # Marginals should match
        torch.testing.assert_close(
            posterior_std.mean, posterior_cached.mean, atol=1e-5, rtol=1e-5
        )

        # Correlation should be computed (may differ slightly due to re-computation)
        assert hasattr(posterior_cached, "correlation_matrix")
```

---

## Part 4: Benchmark Script Fixes

### File: `benchmarks/memory_benchmark_caching.py`

#### 4.1 Fix Gradient Verification (lines 265-300)

Replace the gradient verification section:

```python
# Benchmark 3: Gradient verification
print("\n" + "-"*40)
print("Benchmark 3: Gradient verification")
print("-"*40)

# Use consistent test input shape
test_shape = (1, n_test, n_features)

# Standard gradient
model.cache_trainset_representation = False
test_x_std = torch.randn(*test_shape, device=device, dtype=dtype, requires_grad=True)
output_std = model(x=train_x, y=train_y, test_x=test_x_std)
output_std.sum().backward()
grad_std = test_x_std.grad.clone()

# Cached gradient - use SAME shape for cache population
model.cache_trainset_representation = True
model.empty_trainset_representation_cache()

# Populate cache with dummy of same shape as actual test
dummy_test = torch.zeros(*test_shape, device=device, dtype=dtype)
with torch.no_grad():
    _ = model(x=train_x, y=train_y, test_x=dummy_test)

# Now compute gradient with cache
test_x_cached = test_x_std.detach().clone().requires_grad_(True)
output_cached = model(x=None, y=None, test_x=test_x_cached)
output_cached.sum().backward()
grad_cached = test_x_cached.grad.clone()

model.empty_trainset_representation_cache()
model.cache_trainset_representation = False

# Compare with tight tolerance
grad_diff = (grad_std - grad_cached).abs().max().item()
output_diff = (output_std.detach() - output_cached.detach()).abs().max().item()

print(f"  Max gradient difference: {grad_diff:.2e}")
print(f"  Max output difference: {output_diff:.2e}")

TOLERANCE = 1e-5
if grad_diff < TOLERANCE and output_diff < TOLERANCE:
    print(f"  ✓ Gradients and outputs match (tolerance: {TOLERANCE})")
else:
    print(f"  ✗ Differences exceed tolerance {TOLERANCE}")
```

### File: `benchmarks/botorch_memory_benchmark.py`

#### 4.2 Tighten Consistency Tolerances

In `verify_prediction_consistency()` (around line 310), update:

```python
STRICT_TOLERANCE = 1e-6

if mean_diff < STRICT_TOLERANCE and var_diff < STRICT_TOLERANCE:
    print(f"\n✓ Predictions are numerically identical (diff < {STRICT_TOLERANCE})")
    return True
elif mean_diff < 1e-4 and var_diff < 1e-4:
    print(f"\n⚠ Predictions differ slightly (mean: {mean_diff:.2e}, var: {var_diff:.2e})")
    print("  This may indicate minor numerical differences in code paths.")
    return True
else:
    print(f"\n✗ Predictions differ significantly - there may be a bug")
    return False
```

#### 4.3 Add Cache Verification

In `run_botorch_benchmark()`, after entering the cache context:

```python
with model.cache_training_context():
    # Verify cache is actually engaged
    assert getattr(model, "_training_cache_ready", False), (
        "Cache not enabled - benchmark is invalid"
    )

    acqf_cached = ExpectedImprovement(model, best_f=best_f)
    # ... rest of optimization
```

---

## Part 5: Implementation Order

Execute in this order to minimize risk:

### Phase 0: Verify Correctness (BLOCKING)

0. **Create and run correctness tests** (MUST PASS BEFORE CONTINUING)
   - Create `tests/test_current_caching_correctness.py` with tests from Part 0
   - Run: `pytest tests/test_current_caching_correctness.py -v`
   - If any tests fail, fix bugs before proceeding
   - Document any bugs found

### Phase 1: Refactoring (only after Phase 0 passes)

1. **Add state initialization and capability checks** (low risk)
   - Add `_training_cache_ready`, `_cached_negate_train_ys`, `_cached_context_hash` to `__init__`
   - Add capability validation to `cache_training_context`

2. **Add helper methods** (low risk)
   - `_compute_context_hash`
   - `_validate_posterior_args`
   - `_check_cache_compatibility`
   - `_validate_cache_context`

3. **Add template method hooks** (medium risk)
   - `_augment_training_data`
   - `_build_posterior`

4. **Refactor `PFNModel.posterior()`** (medium risk)
   - Replace with template method implementation
   - Update `_prepare_data` to return `None`
   - Update `pfn_predict` with explicit `use_cache` param

5. **Refactor subclasses** (medium risk)
   - `PFNModelWithPendingPoints` → use `_augment_training_data` hook
   - `MultivariatePFNModel` → use `_build_posterior` hook

6. **Update `cache_training_context`** (medium risk)
   - Add `pending_X` parameter
   - Add context hash computation

7. **Add tests** (low risk)
   - Create `tests/test_botorch_pfn_caching.py`
   - Run to verify all behavior

8. **Fix benchmarks** (low risk)
   - Fix gradient verification
   - Tighten tolerances
   - Add cache verification

---

## Validation Checklist

### Phase 0 Checklist (must complete first)

- [ ] `test_current_caching_correctness.py` created
- [ ] All `TestCachingNumericalEquivalence` tests pass
- [ ] All `TestCachingGradientCorrectness` tests pass
- [ ] All `TestCachingEdgeCases` tests pass
- [ ] All `TestTransformerLevelCaching` tests pass
- [ ] Any bugs found are documented and fixed
- [ ] Re-run all tests after fixes to confirm

### Phase 1+ Checklist (after refactoring)

- [ ] `PFNModel` cache state is initialized in `__init__`
- [ ] `cache_training_context` raises on incompatible transformers
- [ ] Cached predictions match non-cached (tolerance < 1e-5)
- [ ] Gradients flow through test_x when caching
- [ ] Gradients match between cached and non-cached (tolerance < 1e-4)
- [ ] `negate_train_ys` mismatch raises `ValueError`
- [ ] `PFNModelWithPendingPoints` works with caching
- [ ] `pending_X` modification blocked during active cache
- [ ] `MultivariatePFNModel` marginals use cache
- [ ] Context hash validation catches changed training context
- [ ] All tests pass
- [ ] Benchmarks pass with fixed validation
