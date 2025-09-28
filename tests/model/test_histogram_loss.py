import torch
import torch.nn as nn

from pfns.model.criterions import HistogramLoss, HistogramLossConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.train import compute_losses


def test_histogram_loss_prefers_close_positives():
    criterion = HistogramLoss(embedding_dim=3, num_bins=10, lower=-1.0, upper=1.0)

    embeddings_worse = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    embeddings_better = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.99, 0.05, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32
    )
    labels = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    loss_worse = criterion(embeddings_worse, labels).mean()
    loss_better = criterion(embeddings_better, labels).mean()

    assert loss_better < loss_worse


def _get_decoder_out_features(decoder: nn.Module) -> int:
    if isinstance(decoder, nn.Sequential):
        return decoder[-1].out_features
    if hasattr(decoder, "out_features"):
        return int(decoder.out_features)
    raise AssertionError("Decoder does not expose an out_features attribute")


def test_histogram_loss_transformer_smoke():
    config = TransformerConfig(
        criterion=HistogramLossConfig(
            embedding_dim=6,
            num_bins=5,
            lower=-1.0,
            upper=1.0,
        ),
        emsize=12,
        nhid=24,
        nlayers=1,
        nhead=2,
    )
    model = config.create_model()

    decoder = model.decoder_dict["standard"]
    assert _get_decoder_out_features(decoder) == 6

    batch_size = 2
    seq_len_train = 3
    seq_len_test = 3
    num_features = 4

    train_x = torch.randn(batch_size, seq_len_train, num_features)
    train_y = torch.randn(batch_size, seq_len_train, 1)
    test_x = torch.randn(batch_size, seq_len_test, num_features)

    output = model(x=train_x, y=train_y, test_x=test_x)

    targets = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 1.0, float("nan")]], dtype=torch.float32
    )

    model.zero_grad()
    losses = compute_losses(output, targets, model.criterion, n_targets_per_input=1)
    loss = losses.mean()

    loss.backward()

    assert losses.shape == (batch_size, seq_len_test)
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
