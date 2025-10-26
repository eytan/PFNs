import torch

from pfns.priors import multitask_regression


def test_multitask_prior_shapes():
    batch = multitask_regression.get_batch(
        batch_size=2,
        seq_len=12,
        num_features=4,
        single_eval_pos=7,
        num_tasks=3,
    )

    assert batch.x.shape == (2, 12, 4)
    assert batch.y.shape == (2, 12, 1)
    assert batch.task_indices is not None
    assert batch.num_tasks == 3
    assert batch.task_indices.shape == (2, 12)
    train_task_ids = batch.task_indices[:, :7]
    assert torch.all(train_task_ids >= 0)
    test_task_ids = batch.task_indices[:, 7:]
    assert torch.all(test_task_ids >= -1)
