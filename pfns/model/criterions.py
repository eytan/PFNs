from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from pfns import base_config

from .bar_distribution import BarDistributionConfig


class HistogramLoss(nn.Module):
    """Differentiable histogram loss operating on cosine similarities.

    The implementation mirrors the histogram loss introduced by Ustinova &
    Lempitsky, *Learning Deep Embeddings with Histogram Loss*, NeurIPS 2016.
    Positive and negative pair similarities are softly assigned to histogram
    bins using triangular kernels, yielding smooth empirical similarity
    distributions. Integrating the positive histogram against the tail of the
    negative histogram approximates the probability that a randomly drawn
    negative pair outranks a positive one, so minimising the loss encourages
    tighter positive clusters and separates negatives. As the training loop
    expects a loss per time-step, the per-item scalar loss is broadcast over the
    sequence dimension before returning.

    Notes:
        * ``num_bins`` controls the resolution of the soft histogram. The
          default of 32 bins matches the configuration used in the original
          paper; increase it (e.g. 128–256) for sharper similarity estimates
          when working with distributions such as :class:`BarDistribution` that
          expose many bins.
        * We operate on cosine similarities to make the loss invariant to the
          embedding scale. Embeddings are L2-normalised with
          ``torch.nn.functional.normalize``, which is differentiable and stable
          thanks to the internal ``eps`` argument.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_bins: int = 32,
        lower: float = -1.0,
        upper: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("HistogramLoss requires at least two bins.")
        if upper <= lower:
            raise ValueError("upper must be greater than lower for HistogramLoss.")

        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.lower = lower
        self.upper = upper
        self.eps = eps

        bin_centers = torch.linspace(lower, upper, num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_width = (upper - lower) / (num_bins - 1)

    def _label_equality(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype.is_floating_point:
            return torch.isclose(
                labels.unsqueeze(-1), labels.unsqueeze(-2), atol=1e-6, rtol=1e-6
            )
        return labels.unsqueeze(-1) == labels.unsqueeze(-2)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute histogram loss for a batch of sequence embeddings.

        Args:
            embeddings: Tensor of shape ``(batch, seq_len, embedding_dim)``.
            labels: Tensor of shape ``(batch, seq_len)``.

        Returns:
            Tensor of shape ``(batch, seq_len)`` containing the per-item loss
            broadcast across the sequence length.
        """

        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings to be 3D (batch, seq, dim) but got {embeddings.shape}"
            )
        if labels.ndim != 2:
            raise ValueError(
                f"Expected labels to be 2D (batch, seq) but got {labels.shape}"
            )
        if embeddings.shape[:2] != labels.shape:
            raise ValueError(
                "Embeddings and labels must share batch and sequence dimensions."
            )

        batch, seq_len, _ = embeddings.shape

        # Normalise embeddings to operate on cosine similarities.
        normed_embeddings = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(normed_embeddings, normed_embeddings.transpose(1, 2))
        similarity = similarity.clamp(self.lower, self.upper)

        if labels.dtype.is_floating_point:
            valid_mask = ~torch.isnan(labels)
            safe_labels = labels
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
            safe_labels = labels

        valid_pairs = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
        diagonal = torch.eye(seq_len, device=embeddings.device, dtype=torch.bool)
        valid_pairs = valid_pairs & ~diagonal

        label_comparison = self._label_equality(safe_labels)
        label_equal = label_comparison & valid_pairs
        label_not_equal = (~label_comparison) & valid_pairs

        # Soft assignment to histogram bins via triangular kernels.
        bin_centers = self.bin_centers.to(similarity.dtype)
        bin_width = similarity.new_tensor(self.bin_width)
        eps = similarity.new_tensor(self.eps)
        diff = similarity.unsqueeze(-1) - bin_centers
        bin_weights = torch.clamp(1 - diff.abs() / (bin_width + eps), min=0.0)

        pos_weights = bin_weights * label_equal.unsqueeze(-1)
        neg_weights = bin_weights * label_not_equal.unsqueeze(-1)

        pos_hist = pos_weights.sum(dim=(1, 2))
        neg_hist = neg_weights.sum(dim=(1, 2))

        pos_count = label_equal.sum(dim=(1, 2))
        neg_count = label_not_equal.sum(dim=(1, 2))

        # Normalise histograms; fall back to zeros when no pairs are available.
        pos_hist = torch.where(
            pos_count.unsqueeze(-1) > 0,
            pos_hist / (pos_count.unsqueeze(-1) + self.eps),
            torch.zeros_like(pos_hist),
        )
        neg_hist = torch.where(
            neg_count.unsqueeze(-1) > 0,
            neg_hist / (neg_count.unsqueeze(-1) + self.eps),
            torch.zeros_like(neg_hist),
        )

        neg_tail = torch.cumsum(neg_hist.flip(-1), dim=-1).flip(-1)
        loss_per_item = (pos_hist * neg_tail).sum(dim=-1)

        # Broadcast scalar loss across sequence length for compatibility with callers.
        loss_matrix = loss_per_item.unsqueeze(-1).expand(batch, seq_len)
        return loss_matrix


@dataclass(frozen=True)
class CrossEntropyConfig(base_config.BaseConfig):
    num_classes: int
    reduction: str = "none"

    def get_criterion(self):
        return nn.CrossEntropyLoss(
            reduction=self.reduction, weight=torch.ones(self.num_classes)
        )


@dataclass(frozen=True)
class HistogramLossConfig(base_config.BaseConfig):
    embedding_dim: int
    num_bins: int = 32
    lower: float = -1.0
    upper: float = 1.0
    eps: float = 1e-12

    def get_criterion(self) -> HistogramLoss:
        return HistogramLoss(
            embedding_dim=self.embedding_dim,
            num_bins=self.num_bins,
            lower=self.lower,
            upper=self.upper,
            eps=self.eps,
        )


__all__ = [
    "BarDistributionConfig",
    "CrossEntropyConfig",
    "HistogramLoss",
    "HistogramLossConfig",
]
