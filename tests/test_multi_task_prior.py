import torch

from pfns.priors import multi_task_gp


def test_multi_task_prior_shapes():
    torch.manual_seed(1)
    batch = multi_task_gp.get_batch(
        batch_size=3,
        seq_len=8,
        num_features=5,
        single_eval_pos=4,
        device=torch.device("cpu"),
    )

    assert batch.x.shape == (3, 8, 5)
    assert batch.y.shape == (3, 8, 1)
    assert batch.target_y.shape == (3, 8, 1)
    assert batch.task_partition is not None
    assert batch.task_partition.shape == (3, 8)
    # ensure multiple tasks present in batch
    assert batch.task_partition.unique().numel() > 1
