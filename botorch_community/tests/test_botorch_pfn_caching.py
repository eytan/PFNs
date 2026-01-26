"""Tests for PFNModel caching functionality.

These tests verify the refactored caching implementation including:
- Cache state initialization
- Cache capability validation
- Context manager behavior
- Cached prediction correctness
- Gradient flow through cached predictions
- Negation handling
- Subclass behavior (PFNModelWithPendingPoints, MultivariatePFNModel)
"""

import pytest
import torch
from unittest.mock import MagicMock

from botorch_community.models.prior_fitted_network import (
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
def dtype():
    return torch.float32


@pytest.fixture
def simple_transformer(device, dtype):
    """Create a simple transformer for testing."""
    torch.manual_seed(42)
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
    model = model.to(device=device, dtype=dtype)

    # Add small perturbation to weights to ensure gradient flow
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))

    return model.eval()


@pytest.fixture
def train_data(device, dtype):
    """Create training data for tests."""
    torch.manual_seed(42)
    n_train, n_dims = 20, 5
    train_X = torch.rand(n_train, n_dims, device=device, dtype=dtype)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-8)
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

    def test_raises_on_missing_cache_attribute(self, train_data, device, dtype):
        """Should raise if transformer doesn't support caching."""
        train_X, train_Y = train_data

        # Create mock model without caching support
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        del mock_model.cache_trainset_representation  # Ensure missing
        mock_model.criterion = MagicMock()
        mock_model.criterion.borders = torch.linspace(-3, 3, 11, device=device, dtype=dtype)

        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=mock_model,
            batch_first=True,
        )

        with pytest.raises(TypeError, match="does not support caching"):
            with model.cache_training_context():
                pass

    def test_raises_on_missing_cache_clear_method(self, train_data, device, dtype):
        """Should raise if transformer lacks cache clearing method."""
        train_X, train_Y = train_data

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.cache_trainset_representation = False
        del mock_model.empty_trainset_representation_cache
        mock_model.criterion = MagicMock()
        mock_model.criterion.borders = torch.linspace(-3, 3, 11, device=device, dtype=dtype)

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

    def test_predictions_match(self, simple_transformer, train_data, device, dtype):
        """Cached and non-cached predictions should be identical."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        torch.manual_seed(123)
        test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached
        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()
            var_std = posterior_std.variance.clone()

        # Cached
        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()
                var_cached = posterior_cached.variance.clone()

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(var_std, var_cached, atol=1e-7, rtol=1e-7)

    def test_predictions_match_batched(self, simple_transformer, train_data, device, dtype):
        """Cached predictions should work with batched inputs."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        torch.manual_seed(456)
        test_X = torch.rand(8, 3, train_X.shape[-1], device=device, dtype=dtype)  # (b, q, d)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)

    def test_multiple_calls_in_context(self, simple_transformer, train_data, device, dtype):
        """Multiple posterior calls within cache context should all be consistent."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        torch.manual_seed(789)
        test_X1 = torch.rand(3, train_X.shape[-1], device=device, dtype=dtype)
        test_X2 = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Get non-cached baselines
        with torch.no_grad():
            mean_std1 = model.posterior(test_X1).mean.clone()
            mean_std2 = model.posterior(test_X2).mean.clone()

        # Get cached results
        with model.cache_training_context():
            with torch.no_grad():
                mean_cached1 = model.posterior(test_X1).mean.clone()
                mean_cached2 = model.posterior(test_X2).mean.clone()

        torch.testing.assert_close(mean_std1, mean_cached1, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(mean_std2, mean_cached2, atol=1e-7, rtol=1e-7)


class TestPFNModelCacheGradients:
    """Test gradient flow through cached predictions."""

    def test_gradients_flow_through_test_x(self, simple_transformer, train_data, device, dtype):
        """Gradients should flow through test_x when caching."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        torch.manual_seed(111)
        test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype, requires_grad=True)

        with model.cache_training_context():
            posterior = model.posterior(test_X)
            loss = posterior.mean.sum()
            loss.backward()

        assert test_X.grad is not None
        assert not torch.all(test_X.grad == 0)

    def test_gradients_match_non_cached(self, simple_transformer, train_data, device, dtype):
        """Gradients should match between cached and non-cached."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        torch.manual_seed(222)
        base_test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached gradient
        test_X_std = base_test_X.clone().requires_grad_(True)
        posterior_std = model.posterior(test_X_std)
        posterior_std.mean.sum().backward()
        grad_std = test_X_std.grad.clone()

        # Cached gradient
        test_X_cached = base_test_X.clone().requires_grad_(True)
        with model.cache_training_context():
            posterior_cached = model.posterior(test_X_cached)
            posterior_cached.mean.sum().backward()
        grad_cached = test_X_cached.grad.clone()

        torch.testing.assert_close(grad_std, grad_cached, atol=1e-7, rtol=1e-7)

    def test_gradient_shape_consistency(self, simple_transformer, train_data, device, dtype):
        """Gradients should have correct shape for various input shapes."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_shapes = [
            (5, train_X.shape[-1]),        # (q, d)
            (4, 1, train_X.shape[-1]),     # (b, q=1, d)
            (3, 2, train_X.shape[-1]),     # (b, q, d)
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


class TestPFNModelCacheNegation:
    """Test cache behavior with negate_train_ys."""

    def test_negate_mismatch_raises(self, simple_transformer, train_data, device, dtype):
        """Should raise if negate_train_ys doesn't match cache."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        with model.cache_training_context(negate_train_ys=False):
            with pytest.raises(ValueError, match="negate_train_ys"):
                model.posterior(test_X, negate_train_ys=True)

    def test_negation_cached_correctly(self, simple_transformer, train_data, device, dtype):
        """Negated predictions should match between cached and non-cached."""
        train_X, train_Y = train_data
        model = PFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
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

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)


class TestPFNModelWithPendingPointsCaching:
    """Test caching with pending points."""

    def test_cache_includes_pending_points(self, simple_transformer, train_data, device, dtype):
        """Cache should include pending points."""
        train_X, train_Y = train_data
        pending_X = torch.rand(3, train_X.shape[-1], device=device, dtype=dtype)

        model = PFNModelWithPendingPoints(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            pending_X=pending_X,
            batch_first=True,
        )

        torch.manual_seed(444)
        test_X = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)

        # Non-cached
        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()

        # Cached (should include pending points)
        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)

    def test_pending_modification_blocked_during_cache(
        self, simple_transformer, train_data, device, dtype
    ):
        """Cannot modify pending_X while cache is active."""
        train_X, train_Y = train_data
        pending_X = torch.rand(3, train_X.shape[-1], device=device, dtype=dtype)

        model = PFNModelWithPendingPoints(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            pending_X=pending_X,
            batch_first=True,
        )

        with model.cache_training_context():
            with pytest.raises(RuntimeError, match="Cannot modify pending_X"):
                model.pending_X = torch.rand(2, train_X.shape[-1], device=device, dtype=dtype)

    def test_pending_points_property(self, simple_transformer, train_data, device, dtype):
        """Test pending_X property getter and setter."""
        train_X, train_Y = train_data
        pending_X = torch.rand(3, train_X.shape[-1], device=device, dtype=dtype)

        model = PFNModelWithPendingPoints(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            pending_X=pending_X,
            batch_first=True,
        )

        # Test getter
        assert model.pending_X is not None
        assert model.pending_X.shape == pending_X.shape

        # Test setter (outside cache context)
        new_pending = torch.rand(5, train_X.shape[-1], device=device, dtype=dtype)
        model.pending_X = new_pending
        assert model.pending_X.shape == new_pending.shape

        # Test setter to None
        model.pending_X = None
        assert model.pending_X is None


class TestMultivariatePFNModelCaching:
    """Test caching with multivariate posteriors."""

    def test_marginals_use_cache(self, simple_transformer, train_data, device, dtype):
        """Marginal computation should use cache."""
        train_X, train_Y = train_data
        model = MultivariatePFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        # Single point - should return BoundedRiemannPosterior
        torch.manual_seed(555)
        test_X = torch.rand(1, train_X.shape[-1], device=device, dtype=dtype)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()

        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)

    def test_multivariate_with_cache(self, simple_transformer, train_data, device, dtype):
        """Multivariate posterior should work with caching."""
        train_X, train_Y = train_data
        model = MultivariatePFNModel(
            train_X=train_X,
            train_Y=train_Y,
            model=simple_transformer,
            batch_first=True,
        )

        # Multiple points - should return MultivariateRiemannPosterior
        torch.manual_seed(666)
        test_X = torch.rand(3, train_X.shape[-1], device=device, dtype=dtype)

        with torch.no_grad():
            posterior_std = model.posterior(test_X)
            mean_std = posterior_std.mean.clone()

        with model.cache_training_context():
            with torch.no_grad():
                posterior_cached = model.posterior(test_X)
                mean_cached = posterior_cached.mean.clone()

        # Marginals should match
        torch.testing.assert_close(mean_std, mean_cached, atol=1e-7, rtol=1e-7)

        # Correlation should be computed
        assert hasattr(posterior_cached, "correlation_matrix")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
