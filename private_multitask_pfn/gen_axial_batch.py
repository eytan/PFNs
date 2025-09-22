import torch
from PFNs.pfns.priors import Batch


default_device = "cuda" if torch.cuda.is_available() else "cpu"

# batch_size: number of datasets
# seq_len: number of training samples per dataset
def axial_train_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    """
    Generate x_i = [x_1, x_2, ..., x_d] and y_i = x_1 + x_2^2 + ... + x_d^d
    """
    x = torch.rand(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    
    for feature in range(num_features):
        y += torch.pow(x[:, :, feature:feature+1], feature + 1)
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
    )


def axial_test_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    x = torch.rand(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    
    for feature in range(num_features):
        y += torch.pow(x[:, :, feature:feature+1], num_features - feature)
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
    )


# batch_size: number of datasets
# seq_len: number of training samples per dataset
def multitask_line_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    max_num_tasks: int,
    num_tasks: int,
    lengthscale: float,
    hyperparameters=None,
    device: str = default_device,
    **kwargs,
):
    # permuted columns
    x = torch.rand(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    task_id = torch.randint(0, num_tasks, (batch_size, seq_len, 1), device=device)
    
    for task in range(num_tasks):
        for feature in range(num_features):
            constant = torch.randn(1, device=device)
            y += constant * torch.pow(x[:, :, feature:feature+1], feature + 1) * (task_id == task).float()
            
    # shuffle x columns
    column_order = torch.randperm(num_features, device=device)
    x = x[:, :, column_order]
    
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )


# multitask_line_batch(batch_size=2, seq_len=10, num_features=2, max_num_tasks=2, num_tasks=2, lengthscale=0.1)
# axial_test_batch(batch_size=2, seq_len=3, num_features=2, max_num_tasks=2, num_tasks=2, lengthscale=0.1)