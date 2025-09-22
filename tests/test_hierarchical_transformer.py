import pytest
import torch

from pfns.model.transformer import TableTransformer


def test_hierarchical_forward_shape_and_gradients():
    torch.manual_seed(0)
    model = TableTransformer(
        ninp=16,
        nhid=32,
        nlayers=2,
        nhead=4,
        features_per_group=1,
        use_task_hierarchy=True,
    )

    batch_size = 2
    seq_len = 6
    single_eval_pos = 3
    num_features = 4

    x = torch.randn(batch_size, seq_len, num_features, requires_grad=True)
    y = torch.randn(batch_size, single_eval_pos, 1)
    task_partition = torch.tensor(
        [[0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 2, 2]],
        dtype=torch.long,
    )

    output = model(
        x=x,
        y=y,
        task_partition=task_partition,
        only_return_standard_out=True,
    )

    assert output.shape[0] == batch_size
    assert output.shape[1] == seq_len - single_eval_pos
    assert output.shape[2] == 1

    output.sum().backward()
    assert x.grad is not None


def test_hierarchical_requires_partition():
    model = TableTransformer(
        ninp=8,
        nhid=16,
        nlayers=1,
        nhead=2,
        features_per_group=1,
        use_task_hierarchy=True,
    )
    x = torch.randn(1, 4, 3)
    y = torch.randn(1, 2, 1)
    with pytest.raises(ValueError):
        model(x=x, y=y, only_return_standard_out=True)

