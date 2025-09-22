import torch
from PFNs.pfns.priors import Batch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

    
default_device = "cuda" if torch.cuda.is_available() else "cpu"

# batch_size: number of datasets
# seq_len: number of training samples per dataset
def task_invariant_batch(
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
    assert num_tasks == 2, "num_tasks must be 2 for task_invariant_batch"
    x = torch.rand(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    task_id = torch.randint(0, num_tasks, (batch_size, seq_len, 1), device=device)
    
    # task 1 is always 0
    # task 1: y = x_1 + x_2^2 + ... + x_d^d
    # task_one_mask = task_id == 0
    # for feature in range(num_features):
    #     y += torch.pow(x[:, :, feature:feature+1], feature + 1) * task_one_mask
    
    # task 2: y = x_1^d + x_2^(d-1) + ... + x_d
    if torch.rand((1,)) < 0.5:
        task_two_mask = task_id == 1
    else:
        task_two_mask = task_id == 0
    for feature in range(num_features):
        y += torch.pow(x[:, :, feature:feature+1], num_features - feature) * task_two_mask
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    

# opposite of task_invariant_batch
def task_invariant_eval_batch(
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
    assert num_tasks == 2, "num_tasks must be 2 for task_invariant_batch"
    x = torch.rand(batch_size, seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    task_id = torch.zeros((batch_size, seq_len, 1), device=device)
    
    # task 2: y = x_1 + x_2^2 + ... + x_d^d
    task_two_mask = task_id == 1
    for feature in range(num_features):
        y += torch.pow(x[:, :, feature:feature+1], feature + 1) * task_two_mask
    
    # task 1: y = x_1^d + x_2^(d-1) + ... + x_d
    task_one_mask = task_id == 0
    for feature in range(num_features):
        y += torch.pow(x[:, :, feature:feature+1], num_features - feature) * task_one_mask
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    
class BatchedMultidimensionalExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y.shape[0]]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]])),
            batch_shape=torch.Size([train_y.shape[0]])
        )
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]]))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def generate_correlated_gp_samples(x, batch_size, base_samples, correlations):
    """
    Generate multiple GP samples with specified correlations to a base sample
    
    Args:
        x: Input tensor of shape (n_points, num_features)
        base_sample: Optional base GP sample to correlate with
        correlations: List of correlation values
    """
    n_points = x.shape[0]
    n_features = x.shape[1]
    
    # Expand x to include batch dimension
    x_batched = x.unsqueeze(0).expand(batch_size, -1, -1)
    
    tolerance = 1e-6
    while tolerance < 1e-3:
        try:
            # Initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([batch_size]))
            model = BatchedMultidimensionalExactGPModel(x_batched, torch.zeros(batch_size, n_points), likelihood).to(x.device)
            
            # Set the kernel parameters
            model.covar_module.base_kernel.lengthscale = torch.ones(batch_size, n_features).to(x) * 0.2
            model.covar_module.outputscale = torch.ones(batch_size).to(x)
            
            # Get the kernel matrices and their Cholesky decompositions
            K = model.covar_module(x_batched).evaluate() + torch.eye(n_points).to(x.device) * tolerance
            L = torch.linalg.cholesky(K)
        
            break
        except Exception as e:
            print("Cholesky decomposition failed, retrying...")
            tolerance *= 10
            # raise e
    
    # Generate the base GP samples if not provided
    if base_samples is None:
        z_base = torch.randn(batch_size, n_points).to(x.device)
        base_samples = torch.bmm(L, z_base.unsqueeze(-1)).squeeze(-1)
    
    # Transform base samples back to standard normal
    z_base = torch.linalg.solve(L, base_samples.unsqueeze(-1)).squeeze(-1)
    
    # Generate correlated samples
    correlated_samples = []
    for correlation in correlations:
        z_new = correlation * z_base + torch.sqrt(torch.tensor(1 - correlation**2)).to(x.device) * torch.randn(batch_size, n_points).to(x.device)
        f_new = torch.bmm(L, z_new.unsqueeze(-1)).squeeze(-1)
        correlated_samples.append(f_new)
    
    return base_samples, correlated_samples

    
def task_corr_batch(
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
    x = torch.rand(seq_len, num_features, device=device)
    y = torch.zeros(batch_size, seq_len, 1, device=device)
    task_id = torch.randint(0, num_tasks, (seq_len, 1), device=device).expand(batch_size, -1, -1)
    
    correlations = torch.linspace(0.95, 0.0, max_num_tasks - 1)
    if hyperparameters.get("permute_tasks", False):
        correlations = np.random.permutation(correlations)
        
    base_sample = None
    base_sample, correlated_samples = generate_correlated_gp_samples(x, batch_size, base_sample, correlations)

    all_samples = [base_sample] + correlated_samples
    
    for i, samples in enumerate(all_samples):
        task_mask = task_id == i
        # print(y.shape, samples.shape, task_mask.squeeze().shape)
        y += samples.unsqueeze(-1) * (task_mask)
        
    x = x.unsqueeze(0).expand(batch_size, -1, -1)
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    

        
        
    return Batch(
        x=x.transpose(0, 1),
        y=y.transpose(0, 1),
        target_y=y.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )
    
    
# main
if __name__ == "__main__":
    import torch
    import gpytorch
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import product

    # Set random seed for reproducibility
    torch.manual_seed(1)

    def generate_grid_inputs(num_points_per_dim, num_features):
        """Generate a grid of input points across multiple dimensions"""
        # Create linearly spaced points for each dimension
        points_per_dim = [torch.linspace(0, 1, num_points_per_dim) for _ in range(num_features)]
        
        # Create a meshgrid of all points
        grid_points = torch.tensor(list(product(*points_per_dim)))
        
        return grid_points


    def plot_2d_samples(x, samples, correlations, title):
        """Plot samples for 2D inputs"""
        num_points = int(np.sqrt(x.shape[0]))
        X = x[:, 0].reshape(num_points, num_points)
        Y = x[:, 1].reshape(num_points, num_points)
        
        fig = plt.figure(figsize=(15, 3*((len(samples)+1)//3 + 1)))
        
        # Plot base sample
        ax = fig.add_subplot(((len(samples)+1)//3 + 1), 3, 1, projection='3d')
        Z = samples[0].reshape(num_points, num_points)
        surf = ax.plot_surface(X.detach(), Y.detach(), Z.detach(), cmap='viridis')
        ax.set_title('Original GP Draw')
        fig.colorbar(surf, ax=ax)
        
        # Plot correlated samples
        for i, (sample, corr) in enumerate(zip(samples[1:], correlations)):
            ax = fig.add_subplot(((len(samples)+1)//3 + 1), 3, i+2, projection='3d')
            Z = sample.reshape(num_points, num_points)
            surf = ax.plot_surface(X.detach(), Y.detach(), Z.detach(), cmap='viridis')
            ax.set_title(f'Correlation: {corr}')
            fig.colorbar(surf, ax=ax)
        
        plt.tight_layout()
        plt.savefig("correlated_gp_samples.png")

    # Parameters
    num_features = 2  # Number of input dimensions
    num_points_per_dim = 20  # Number of points per dimension
    correlations = [0.9, 0.7, 0.5, 0.3, 0.1]

    # Generate input points
    x = generate_grid_inputs(num_points_per_dim, num_features)

    # Generate samples
    base_sample = None
    base_sample, correlated_samples = generate_correlated_gp_samples(x, 1, base_sample, correlations)

    # Combine all samples for plotting
    all_samples = [base_sample] + correlated_samples

    # Plot results
    if num_features == 2:
        plot_2d_samples(x, all_samples, correlations, 'GP Samples with Different Correlations')
        
    # Print achieved correlations
    for i, corr in enumerate(correlations):
        empirical_corr = np.corrcoef(base_sample.detach().numpy(), correlated_samples[i].detach().numpy())[0,1]
        print(f"Target correlation: {corr:.2f}, Achieved correlation: {empirical_corr:.2f}")
        
    