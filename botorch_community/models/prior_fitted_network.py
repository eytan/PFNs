#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This module defines the botorch model for PFNs (``PFNModel``) and it
provides handy helpers to download pretrained, public PFNs
with ``download_model`` and model paths with ``ModelPaths``.
For the latter to work ``pfns4bo`` must be installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.logging import logger
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import match_batch_shape
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from botorch_community.posteriors.riemann import (
    BoundedRiemannPosterior,
    MultivariateRiemannPosterior,
)
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from pfns.train import MainConfig  # @manual=//pytorch/PFNs:PFNs
from torch import Tensor
from torch.nn import Module


def get_styles(
    model: Module, hps: dict | None, batch_size: int, device: str
) -> dict[str, Tensor]:
    if hps is None or (model.style_encoder is None and model.y_style_encoder is None):
        return {}
    style_kwargs = {}
    if model.style_encoder is not None:
        hps_subset = {
            k: v for k, v in hps.items() if k in model.style_encoder[0].hyperparameters
        }
        style = (
            model.style_encoder[0]
            .hyperparameters_dict_to_tensor(hps_subset)
            .repeat(batch_size, 1)
            .to(device)
            .float()
        )  # shape (batch_size, num_styles)
        style_kwargs["style"] = style

    if model.y_style_encoder is not None:
        hps_subset = {
            k: v
            for k, v in hps.items()
            if k in model.y_style_encoder[0].hyperparameters
        }
        y_style = (
            model.y_style_encoder[0]
            .hyperparameters_dict_to_tensor(hps_subset)
            .repeat(batch_size, 1)
            .to(device)
            .float()
        )  # shape (batch_size, num_styles)
        style_kwargs["y_style"] = y_style
    return style_kwargs


class PFNModel(Model):
    """Prior-data Fitted Network"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        model: Module | None = None,
        checkpoint_url: str = ModelPaths.pfns4bo_hebo,
        train_Yvar: Tensor | None = None,
        batch_first: bool = False,
        constant_model_kwargs: dict[str, Any] | None = None,
        input_transform: InputTransform | None = None,
        load_training_checkpoint: bool = False,
        style_hyperparameters: dict[str, Any] | None = None,
        style: Tensor
        | None = None,  # should have shape (num_styles,) or (num_features, num_styles)
    ) -> None:
        """Initialize a PFNModel.

        Either a pre-trained PFN model can be provided via the model kwarg,
        or a checkpoint_url can be provided from which the model will be
        downloaded. This defaults to the pfns4bo_hebo model.

        Loading the model does an unsafe "weights_only=False" load, so
        it is essential that checkpoint_url be a trusted source.

        Args:
            train_X: A ``n x d`` tensor of training features.
            train_Y: A ``n x 1`` tensor of training observations.
            model: A pre-trained PFN model with the following
                forward(train_X, train_Y, X) -> logit predictions of shape
                ``n x b x c`` where c is the number of discrete buckets
                borders: A ``c+1``-dim tensor of bucket borders.
            checkpoint_url: The string URL of the PFN model to download and load.
                Will be ignored if model is provided.
            train_Yvar: Observed variance of train_Y. Currently ignored.
            batch_first: Whether the batch dimension is the first dimension of
                the input tensors. This is needed to support different PFN
                models. For batch-first x has shape ``batch x seq_len x features``
                and for non-batch-first it has shape ``seq_len x batch x features``.
            constant_model_kwargs: A dictionary of model kwargs that
                will be passed to the model in each forward pass.
            input_transform: A Botorch input transform.
            load_training_checkpoint: Whether to load a training checkpoint as
                produced by the PFNs training code, see github.com/automl/PFNs.
            style_hyperparameters: A dictionary of hyperparameters to be passed
                to the style and the y-style encoders. It is useful when training
                models with ``hyperparameter_sampling`` prior and its style
                encoder. One simply supplies the dict with the unnormalized
                hyperparameters, e.g., {"noise_std": 0.1}. Omitted values are
                treated as unknown and the value will build a Bayesian average
                for these, if ``hyperparameter_sampling_skip_style_prob`` > 0
                during pre-training.
            style: A tensor of style values to be passed to the model. These
                are raw style values of shape (num_styles,), which will then
                be extended as needed.

        """
        super().__init__()
        if model is None:
            model = download_model(
                model_path=checkpoint_url,
            )

        if load_training_checkpoint:
            # the model is not an actual model, but a training checkpoint
            # make a model out of it
            checkpoint = model
            config = MainConfig.from_dict(checkpoint["config"])
            model = config.model.create_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        if train_Yvar is not None:
            logger.debug("train_Yvar provided but ignored for PFNModel.")

        if train_Y.dim() != 2:
            raise UnsupportedError("train_Y must be 2-dimensional.")

        if train_X.dim() != 2:
            raise UnsupportedError("train_X must be 2-dimensional.")

        if train_Y.shape[-1] > 1:
            raise UnsupportedError("Only 1 target allowed for PFNModel.")

        if train_X.shape[0] != train_Y.shape[0]:
            raise UnsupportedError(
                "train_X and train_Y must have the same number of rows."
            )

        with torch.no_grad():
            self.transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self.train_X = train_X  # shape: (n, d)
        self.train_Y = train_Y  # shape: (n, 1)
        # Downstream botorch tooling expects a likelihood to be specified,
        # so here we use a FixedNoiseGaussianLikelihood that is unused.
        if train_Yvar is None:
            train_Yvar = torch.zeros_like(train_Y)
        self.train_Yvar = train_Yvar  # shape: (n, 1)
        self.likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        self.pfn = model.to(device=train_X.device)
        self.batch_first = batch_first
        self.constant_model_kwargs = constant_model_kwargs or {}
        self.style_hyperparameters = style_hyperparameters
        self.style = style
        if input_transform is not None:
            self.input_transform = input_transform
        self._compute_styles()

        # Cache state initialization
        self._training_cache_ready = False
        self._cached_negate_train_ys = False
        self._cached_context_hash: int | None = None
        self._cached_y_encoder_state: dict | None = None

    def _compute_styles(self):
        """
        Can be used to compute styles to be used for PFN prediction based on
        training data.

        When implemented, will directly modify self.style_hyperparameters or
        self.style.
        """
        pass

    def _compute_context_hash(
        self, train_X: Tensor, train_Y: Tensor, negate_train_ys: bool
    ) -> int:
        """Compute hash of training context to detect changes.

        Returns a hash based on training data and negation setting.
        Used to validate cache consistency, especially with pending points.
        """
        return hash((
            id(train_X),
            train_X.shape,
            tuple(train_X.stride()),
            id(train_Y),
            train_Y.shape,
            tuple(train_Y.stride()),
            negate_train_ys,
        ))

    @contextmanager
    def cache_training_context(
        self, negate_train_ys: bool = False
    ) -> Iterator[None]:
        """Context manager to cache training data for efficient batch evaluation.

        When used with BoTorch's ``optimize_acqf()``, this caches the training
        data representations once, allowing many test point evaluations without
        recomputing the training context. Gradients flow through test_x only.

        Memory savings: O(n) instead of O(b × n) for training data, where
        b is the batch size (num_restarts) and n is the training set size.

        Usage::

            with model.cache_training_context():
                acqf = ExpectedImprovement(model, best_f=train_Y.max())
                candidates, value = optimize_acqf(
                    acqf, bounds=bounds,
                    num_restarts=64,
                    raw_samples=512,
                    options={"batch_limit": 64}
                )

        Args:
            negate_train_ys: Whether to negate training Y values (for minimization).
        """
        # Store original state
        original_cache_trainset_representation = getattr(
            self.pfn, "cache_trainset_representation", False
        )

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

        try:
            # Enable caching on the transformer
            self.pfn.cache_trainset_representation = True

            # Prepare training data (single batch)
            train_X_bf = self.transformed_X.unsqueeze(0)  # (1, n, d)
            train_Y_bf = self.train_Y.unsqueeze(0)  # (1, n, 1)
            if negate_train_ys:
                assert self.train_Y.mean().abs() < 1e-4, (
                    "train_Y must be zero-centered for negation."
                )
                train_Y_bf = -train_Y_bf

            # Hook for subclasses to augment training data (e.g., pending points)
            dummy_X = torch.zeros(
                1, 1, train_X_bf.shape[-1],
                device=train_X_bf.device,
                dtype=train_X_bf.dtype,
            )
            train_X_bf, train_Y_bf = self._augment_training_data(
                train_X_bf, train_Y_bf, dummy_X,
                negate_train_ys=negate_train_ys, use_cache=False
            )

            # Get styles for batch_size=1 (will be broadcast during inference)
            styles = self._get_styles(batch_size=1)

            # Populate cache with training data (no gradients on train)
            with torch.no_grad():
                if not self.batch_first:
                    train_X_bf = train_X_bf.transpose(0, 1)  # (n, 1, d)
                    train_Y_bf = train_Y_bf.transpose(0, 1)  # (n, 1, 1)
                    # Create dummy test point to trigger full forward
                    dummy_test = torch.zeros(
                        1, 1, train_X_bf.shape[-1],
                        device=train_X_bf.device,
                        dtype=train_X_bf.dtype,
                    )  # (1, 1, d) -> seq-first
                else:
                    dummy_test = torch.zeros(
                        1, 1, train_X_bf.shape[-1],
                        device=train_X_bf.device,
                        dtype=train_X_bf.dtype,
                    )  # (1, 1, d) -> batch-first

                # Forward pass to populate cache
                self.pfn(
                    x=train_X_bf.float(),
                    y=train_Y_bf.float(),
                    test_x=dummy_test.float(),
                    **self.constant_model_kwargs,
                    **styles,
                )

            # Save y_encoder state after cache population.
            # This is critical for MultivariatePFNModel where _compute_conditional_means()
            # temporarily disables caching and calls the encoder with augmented data,
            # which corrupts the encoder state. We save the state here so it can be
            # restored after each correlation estimation call.
            if hasattr(self.pfn, "y_encoder") and hasattr(
                self.pfn.y_encoder, "save_fitted_state"
            ):
                self._cached_y_encoder_state = self.pfn.y_encoder.save_fitted_state()
            else:
                self._cached_y_encoder_state = None

            # Mark cache as ready and store context hash
            self._training_cache_ready = True
            self._cached_negate_train_ys = negate_train_ys
            self._cached_context_hash = self._compute_context_hash(
                train_X_bf, train_Y_bf, negate_train_ys
            )

            yield

        finally:
            # Clear cache and restore state
            self._training_cache_ready = False
            self._cached_negate_train_ys = False
            self._cached_context_hash = None
            self._cached_y_encoder_state = None

            self.pfn.cache_trainset_representation = original_cache_trainset_representation
            if hasattr(self.pfn, "empty_trainset_representation_cache"):
                self.pfn.empty_trainset_representation_cache()

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

    def _augment_training_data(
        self,
        train_X: Tensor | None,
        train_Y: Tensor | None,
        X: Tensor,
        *,
        negate_train_ys: bool,
        use_cache: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Hook for subclasses to augment training data.

        Override this instead of posterior() to add pending points, etc.

        Args:
            train_X: Training features, shape (b, n, d) or None if use_cache
            train_Y: Training targets, shape (b, n, 1) or None if use_cache
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
        train_X: Tensor | None,
        train_Y: Tensor | None,
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
        orig_X_shape = X.shape  # X has shape b? x q? x d
        if len(X.shape) > 3:
            raise UnsupportedError(f"X must be at most 3-d, got {X.shape}.")
        while len(X.shape) < 3:
            X = X.unsqueeze(0)

        X = self.transform_inputs(X)  # shape (b , q, d)

        if skip_train_replication:
            # When using cached training context, we don't need to replicate
            # training data since it's already cached in the transformer.
            train_X = None
            train_Y = None
        else:
            train_X = match_batch_shape(self.transformed_X, X)  # shape (b, n, d)
            train_Y = match_batch_shape(self.train_Y, X)  # shape (b, n, 1)
            if negate_train_ys:
                assert self.train_Y.mean().abs() < 1e-4, "train_Y must be zero-centered."
                train_Y = -train_Y

        styles = self._get_styles(
            batch_size=X.shape[0],
        )  # shape (b, num_styles)
        return X, train_X, train_Y, orig_X_shape, styles

    def _get_styles(self, batch_size) -> dict[str, Tensor]:
        style_kwargs = get_styles(
            model=self.pfn,
            hps=self.style_hyperparameters,
            batch_size=batch_size,
            device=self.train_X.device,
        )
        if self.style is not None:
            assert style_kwargs == {}, (
                "Cannot provide both style and style_hyperparameters."
            )
            style_kwargs["style"] = (
                self.style[None]
                .repeat(batch_size, 1, 1)
                .to(self.train_X.device)
                .float()
            )
        return style_kwargs

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
            assert self._training_cache_ready, (
                "Cache requested but not populated. Call cache_training_context() first."
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

    @property
    def borders(self):
        return self.pfn.criterion.borders.to(self.train_X.dtype)

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model (always 1 for PFNModel)."""
        return 1


class PFNModelWithPendingPoints(PFNModel):
    """PFNModel that supports pending points (unobserved evaluations).

    Pending points are added to the training context with NaN labels,
    allowing the model to account for in-flight evaluations during
    Bayesian optimization.
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
        """Set pending points. Cannot modify while cache is active."""
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
        """Add pending points to training data.

        Args:
            train_X: Training features (b, n, d) or None if using cache
            train_Y: Training targets (b, n, 1) or None if using cache
            X: Test points (b, q, d)
            negate_train_ys: Whether Y values are negated
            use_cache: Whether using cached training context

        Returns:
            Augmented (train_X, train_Y) or (None, None) if using cache
        """
        if self._pending_X is None:
            return train_X, train_Y

        if use_cache:
            # When using cache, pending points were included at cache creation
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


class MultivariatePFNModel(PFNModel):
    """A multivariate PFN model that returns a joint posterior over q batch inputs.

    For this to work correctly it is necessary that the underlying model return a
    posterior for the latent f, not the noisy observed y.
    """

    def _build_posterior(
        self,
        X: Tensor,
        probabilities: Tensor,
        orig_X_shape: torch.Size,
        train_X: Tensor | None,
        train_Y: Tensor | None,
        styles: dict[str, Tensor],
        use_cache: bool,
    ) -> Union[BoundedRiemannPosterior, MultivariateRiemannPosterior]:
        """Build multivariate posterior with correlation estimation.

        Args:
            X: Prepared test points (b, q, d)
            probabilities: Predicted probabilities (b?, q?, num_buckets)
            orig_X_shape: Original shape of test points
            train_X: Training features or None if using cache
            train_Y: Training targets or None if using cache
            styles: Style tensors for forward pass
            use_cache: Whether using cached training context

        Returns:
            MultivariateRiemannPosterior if q > 1, else BoundedRiemannPosterior
        """
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
            # Apply any augmentation (e.g., pending points from subclasses)
            train_X, train_Y = self._augment_training_data(
                train_X, train_Y, X,
                negate_train_ys=self._cached_negate_train_ys if use_cache else False,
                use_cache=False,  # Force non-cached for correlation
            )

        # Estimate correlation structure with additional forward pass
        R = self.estimate_correlations(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            styles=styles,
            marginals=marginals,
        )  # (b, q, q)
        R = R.view(*orig_X_shape[:-2], X.shape[-2], X.shape[-2])  # (b?, q, q)

        return MultivariateRiemannPosterior(
            borders=self.borders,
            probabilities=marginals.probabilities,
            correlation_matrix=R,
        )

    def estimate_correlations(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        styles: dict[str, Tensor],
        marginals: BoundedRiemannPosterior,
    ) -> Tensor:
        """
        Estimate a correlation matrix R across the q batch of points in X.
        Will do a forward pass through the PFN model with batch size O(q^2).

        For every x_q in [x_1, ..., x_Q]:
           1. Add x_q to train_X, with y_q the 90th percentile value for f(x_q)
           2. Evaluate p(f(x_i)) for all points.

        Uses bivariate normal conditioning formulae, and so will be approximate.

        Args:
            X: evaluation point, shape (b, q, d)
            train_X: Training X, shape (b, n, d)
            train_Y: Training Y, shape (b, n, 1)
            styles: dict from name to tensor shaped (b, ns) for any styles.
            marginals: A posterior object with marginal posteriors for f(X), but no
                correlation structure yet added. posterior.probabilities has
                shape (b?, q, num_buckets).

        Returns: A (b, q, q) correlation matrix
        """
        # Compute conditional distributions with a forward pass
        cond_mean, cond_val = self._compute_conditional_means(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            styles=styles,
            marginals=marginals,
        )
        # Get marginal moments
        var = marginals.variance.squeeze(-1)  # (b?, q)
        mean = marginals.mean.squeeze(-1)  # (b?, q)
        if len(var.shape) == 1:
            var = var.unsqueeze(0)  # (b, q)
            mean = mean.unsqueeze(0)  # (b, q)
        # Estimate covariances from conditional distributions
        cov = self._estimate_covariances(
            cond_mean=cond_mean,
            cond_val=cond_val,
            mean=mean,
            var=var,
        )
        # Convert to correlation matrix
        S = 1 / torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))  # (b, q)
        S = S.unsqueeze(-1).expand(cov.shape)  # (b, q, q)
        R = S * cov * S.transpose(-1, -2)  # (b, q, q)
        return R

    def _compute_conditional_means(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        styles: dict[str, Tensor],
        marginals: BoundedRiemannPosterior,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute conditional means between pairs of points in X.

        Conditioning is done with an additional forward pass through the model. The
        returned conditional mean will be of shape (b, q, q), with entry [b, i, j] the
        conditional mean of j given i set to the conditioning value.

        Args:
            X: evaluation point, shape (b, q, d)
            train_X: Training X, shape (b, n, d)
            train_Y: Training Y, shape (b, n, 1)
            styles: dict from name to tensor shaped (b, ns) for any styles.
            marginals: A posterior object with marginal posteriors for f(X), but no
                correlation structure yet added. posterior.probabilities has
                shape (b?, q, num_buckets).

        Returns: conditional means (b, q, q), and values used for conditioning (b, q).
        """
        b, q, d = X.shape
        n = train_X.shape[-2]
        post_shape = marginals.probabilities.shape[:-1]
        # Find the 90th percentile of each eval point.
        cond_val = marginals.icdf(
            torch.full(post_shape, 0.9, device=X.device, dtype=X.dtype).unsqueeze(0)
        )  # (1, b?, q, 1)
        cond_val = cond_val.view(b, q)  # (b, q)
        # Construct conditional training data.
        # train_X will have shape (b, q, n+1, d), to have a conditional observation
        # for each point. train_Y will have shape (b, q, n+1, 1).
        train_X = train_X.unsqueeze(1).expand(b, q, n, d)
        cond_X = X.unsqueeze(-2)  # (b, q, 1, d)
        train_X = torch.cat((train_X, cond_X), dim=-2)  # (b, q, n+1, d)
        train_Y = train_Y.unsqueeze(1).expand(b, q, n, 1)
        cond_Y = cond_val.unsqueeze(-1).unsqueeze(-1)  # (b, q, 1, 1)
        train_Y = torch.cat((train_Y, cond_Y), dim=-2)  # (b, q, n+1, 1)
        cond_styles = {}
        for name, style in styles.items():
            ns = style.shape[-1]
            cond_styles[name] = style.unsqueeze(-2).expand(b, q, ns).reshape(b * q, ns)
        # Construct eval points
        eval_X = X.unsqueeze(1).expand(b, q, q, d)
        # Squeeze everything into necessary 2 batch dims, and do PFN forward pass
        # Temporarily disable caching for correlation estimation since we use
        # a different training set (with conditioning point) than the cached one
        cache_was_enabled = self.pfn.cache_trainset_representation
        try:
            self.pfn.cache_trainset_representation = False
            cond_probabilities = self.pfn_predict(
                X=eval_X.reshape(b * q, q, d),
                train_X=train_X.reshape(b * q, n + 1, d),
                train_Y=train_Y.reshape(b * q, n + 1, 1),
                **cond_styles,
            )  # (b * q, q, num_buckets)
        finally:
            self.pfn.cache_trainset_representation = cache_was_enabled
            # Restore y_encoder state if it was saved during cache population.
            # The pfn_predict call above fitted the encoder on augmented data (n+1 points),
            # which corrupts the state that was set when the cache was populated.
            if (
                hasattr(self, "_cached_y_encoder_state")
                and self._cached_y_encoder_state is not None
                and hasattr(self.pfn, "y_encoder")
                and hasattr(self.pfn.y_encoder, "restore_fitted_state")
            ):
                self.pfn.y_encoder.restore_fitted_state(self._cached_y_encoder_state)
        # Object for conditional posteriors
        cond_posterior = BoundedRiemannPosterior(
            borders=self.borders,
            probabilities=cond_probabilities,
        )
        # Get conditional means
        cond_mean = cond_posterior.mean.squeeze(-1)  # (b * q, q)
        cond_mean = cond_mean.unsqueeze(0).view(b, q, q)
        return cond_mean, cond_val

    def _estimate_covariances(
        self,
        cond_mean: Tensor,
        cond_val: Tensor,
        mean: Tensor,
        var: Tensor,
    ) -> Tensor:
        """
        Estimate covariances from conditional distributions.

        Part one: Compute noise variance implied by conditional distributions
        E[f_j | y_j=y] = E[f_j] + var[f_j]/(var[f_j] + noise_var) * (y - E[f_j])
        Let Z_jj = (E[f_j | y_j=y] - E[f_j]) / (y - E[f_j]).
        Note that Z is in (0, 1].
        Then, noise_var_j = var[f_j](1/Z_jj - 1).

        Part two: Compute covariances for all pairs
        E[f_j|y_i=y] = E[f_j]+cov[f_j, f_i]/(var[f_i] + noise_var_i) * (y - E[f_i])
        Let Z_ij = (E[f_j | y_i=y] - E[f_j]) / (y - E[f_i]).
        Then, cov[f_j, f_i] = Z * (var[f_i] + noise_var)

        Args:
            cond_mean: (b, q, q) means of dim -1 conditioned on dim -2
            cond_val: (b, q) conditioned y value.
            var: (b, q) marginal variances
            mean: (b, q) marginal means

        Returns: Covariance matrix
        """
        Z = (cond_mean - mean.unsqueeze(-2).expand(cond_mean.shape)) / (
            cond_val - mean
        ).unsqueeze(-1)  # (b, q, q)
        # Z[i, j] is for j cond. on i
        noise_var = torch.clamp(
            var * (1 / torch.diagonal(Z, dim1=-2, dim2=-1) - 1), min=1e-8
        )  # (b, q)
        cov = Z * (var + noise_var).unsqueeze(-1)  # (b, q, q)
        # Symmetrize
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        cov = self._map_psd(cov)
        return cov

    def _map_psd(self, A):
        """
        Map A (assumed symmetric) to the nearest PSD matrix.
        """
        if torch.linalg.eigvals(A).real.min() < 0:
            L, Q = torch.linalg.eigh(A)
            L = torch.clamp(L, min=1e-6)
            A = Q @ torch.diag_embed(L) @ Q.transpose(-1, -2)
        return A
