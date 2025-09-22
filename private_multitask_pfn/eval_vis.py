import argparse

import botorch
import gpytorch
import torch

from utils import *
import traceback
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal
import os

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot(ax, test_info, pfn_info):
    pfn_id, model, model_dim = pfn_info
    train_x, train_y, test_x, test_y = test_info
    train_task_id = torch.zeros(train_x.size(0))
    
    if model_dim > 1:
        # pad train and test with 0s
        padded_train_x = torch.cat([train_x, torch.zeros(train_x.size(0), model_dim - 1)], dim=1)
        padded_test_x = torch.cat([test_x, torch.zeros(test_x.size(0), model_dim - 1)], dim=1)
    else:
        padded_train_x = train_x
        padded_test_x = test_x
        
    # train_x, train_task_id, train_y, test_x, 
    posterior = pfn_predict(model, train_task_id.to(default_device), padded_train_x.to(default_device), train_y.to(default_device), padded_test_x.to(default_device))
    mean = posterior.mean.cpu()
    std = posterior.variance.sqrt().cpu()
    
    ax.plot(test_x, test_y, label="true", color="C0")
    ax.scatter(train_x, train_y, label="train", color="C0")
    ax.plot(test_x, mean, label="mean", color="C1")
    ax.fill_between(test_x.flatten(), mean.flatten() - 2 * std.flatten(), mean.flatten() + 2 * std.flatten(), alpha=0.2, color="C1")


def get_test_function(seed):
    with botorch.manual_seed(seed):
        n_samples = torch.randint(5, 20, (1,)).item()
        
        train_x = torch.rand(n_samples, 1)
        test_x = torch.linspace(0, 1, 100).view(-1, 1)
        all_x = torch.cat([train_x, test_x], dim=0)
        
        rbf = gpytorch.kernels.RBFKernel()
        # rbf.lengthscale = torch.tensor(0.2)
        rbf.lengthscale = torch.distributions.Gamma(3, 6).sample()
        
        covar = rbf(all_x)
        covar = covar + 1e-4 * torch.eye(covar.size(0))
        
        y = MultivariateNormal(torch.zeros(all_x.size(0)), covar).sample()
        train_y = y[:n_samples]
        test_y = y[n_samples:]
        
    return train_x, train_y, test_x, test_y



def plot_multitask(ax_multi, ax_single, test_info, pfn_info):
    pfn_id, model, model_dim = pfn_info
    train_x, train_task_id, train_y, test_x, test_task_id, test_y = test_info
    
    if model_dim > 1:
        # pad train and test with 0s
        padded_train_x = torch.cat([train_x, torch.zeros(train_x.size(0), model_dim - 1)], dim=1)
        padded_test_x = torch.cat([test_x, torch.zeros(test_x.size(0), model_dim - 1)], dim=1)
    else:
        padded_train_x = train_x
        padded_test_x = test_x
        
    # train_x, train_task_id, train_y, test_x
    posterior = pfn_predict(model, train_task_id.to(default_device), padded_train_x.to(default_device), train_y.to(default_device), padded_test_x.to(default_device))
    mean = posterior.mean.cpu()
    std = posterior.variance.sqrt().cpu()
    nll = -torch.distributions.Normal(mean.squeeze(), std.squeeze()).log_prob(test_y.squeeze()).mean()
    
    ax_multi.set_title(f"Multi NLL: {nll.item():.2f}")
    ax_multi.plot(test_x, test_y, label="true", color="C0")
    ax_multi.plot(test_x, mean, label="mean", color="C1")
    ax_multi.fill_between(test_x.flatten(), mean.flatten() - 2 * std.flatten(), mean.flatten() + 2 * std.flatten(), alpha=0.2, color="C1")
    
    for i in train_task_id.unique():
        mask = train_task_id == i
        marker = "x" if i.item() == 0 else "o"
        size = 100 if i.item() == 0 else 50
        ax_multi.scatter(train_x[mask], train_y[mask], label=f"train task {i.item()}", color=f"C{i}", marker=marker, s=size)


    # train_x, train_task_id, train_y, test_x,
    target_mask = train_task_id == 0
    posterior = pfn_predict(model, train_task_id[target_mask].to(default_device), padded_train_x[target_mask].to(default_device), train_y[target_mask].to(default_device), padded_test_x.to(default_device))
    mean = posterior.mean.cpu()
    std = posterior.variance.sqrt().cpu()
    nll = -torch.distributions.Normal(mean.squeeze(), std.squeeze()).log_prob(test_y.squeeze()).mean()
    
    ax_single.set_title(f"Single NLL: {nll.item():.2f}")
    ax_single.plot(test_x, test_y, label="true", color="C0")
    ax_single.plot(test_x, mean, label="mean", color="C1")
    ax_single.fill_between(test_x.flatten(), mean.flatten() - 2 * std.flatten(), mean.flatten() + 2 * std.flatten(), alpha=0.2, color="C1")
    
    for i in [0]:#train_task_id.unique():
        mask = train_task_id == i
        # marker = "x" if i.item() == 0 else "o"
        # size = 100 if i.item() == 0 else 50
        marker = "x"
        size = 100
        ax_single.scatter(train_x[mask], train_y[mask], label=f"train task {i}", color=f"C{i}", marker=marker, s=size)


def get_multitask_test_function(seed, n_target, n_source):
    n_features = 1
    n_tasks = 2
    with botorch.manual_seed(seed):
        if n_target is not None:
            n_samples = n_target + n_source * (n_tasks - 1)
            target_task_id = torch.zeros(n_target, 1).long()
            source_task_ids = [torch.ones(n_source, 1).long() * i for i in range(1, n_tasks)]
            train_task_id = torch.cat([target_task_id] + source_task_ids, dim=0)
        else:
            n_samples = torch.randint(10, 40, (1,)).item()
            train_task_id = torch.randint(n_tasks, size=(n_samples,)).unsqueeze(1).long()
        
        train_xs = torch.rand(n_samples, n_features)
        test_xs = torch.linspace(0, 1, 100).view(-1, n_features)
        test_task_id = torch.zeros(test_xs.size(0), 1).long()
        xs = torch.cat([train_xs, test_xs], dim=0)
        task_id = torch.cat([train_task_id, test_task_id], dim=0)
        
        rbf = RBFKernel()
        rbf.lengthscale = torch.distributions.Gamma(3, 6).sample()
        covar_x = rbf(xs)
        task_covar_matrix = torch.ones(n_tasks, n_tasks) * 0.9
        task_covar_matrix += torch.eye(n_tasks) * 0.1
        
        covar_t = task_covar_matrix[task_id, task_id.t()].squeeze()
        covar = covar_x.mul(covar_t)
        covar = covar + 1e-4 * torch.eye(covar.size(0))
        ys = MultivariateNormal(torch.zeros(covar.shape[-1]), covar).sample()

        train_y = ys[:n_samples]
        test_y = ys[n_samples:]
        
    return train_xs, train_task_id.squeeze(), train_y, test_xs, test_task_id.squeeze(), test_y


# main function
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # pfn wandb ids
    argparser.add_argument("--pfn_ids", nargs="*", type=str, help="wandb ids for pfn models")
    argparser.add_argument("--dirs", nargs="*", type=str, help="checkpoints for pfn models")
    argparser.add_argument("--seed", type=int, default=0, help="seed for test function")
    argparser.add_argument("--n_target", type=int, default=None, help="samples of target function")
    argparser.add_argument("--n_source", type=int, default=None, help="samples of source function")
    args = argparser.parse_args()
    
    pfn_dirs = []
    if args.pfn_ids is not None:
        # pfn_dirs += ["/home/yl9959/mtpfn/wandb_links/%s" % pfn_id for pfn_id in args.pfn_ids]
        if os.path.exists("/home/lily_l/private_multitask_pfn/wandb_links"):
            pfn_dirs += [(pfn_id, "/home/lily_l/private_multitask_pfn/wandb_links/%s" % pfn_id) for pfn_id in args.pfn_ids]
        else:
            pfn_dirs += [(pfn_id, "/home/yl9959/mtpfn/wandb_links/%s" % pfn_id) for pfn_id in args.pfn_ids]
    if args.dirs is not None:
        pfn_dirs += [("pfn_%d" % i, dir) for i, dir in enumerate(args.dirs)]

    for pfn_id, pfn_dir in pfn_dirs:
        try:
            pfn_args_json = f"{pfn_dir}/args.json"
            with open(pfn_args_json, "r") as f:
                pfn_args = json.load(f)
                
            if not pfn_args["sample_num_features"]:
                model_dim = pfn_args["num_features"]
            else:
                model_dim = 1
            
            model = load_model(pfn_dir, best=True).to(default_device)
            pfn_info = (pfn_id, model, model_dim)
                        
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            
            for i in range(5):
                seed = args.seed + i
                test_info = get_test_function(seed)
                plot(axs[i], test_info, pfn_info)
            
            plt.legend()    
            plt.suptitle(pfn_info[0])
            plt.tight_layout()
            plt.savefig(f"figures/pfn_{pfn_info[0]}.png")
            plt.close()
            
            fig, axs = plt.subplots(2, 5, figsize=(20, 8))
            # axs = axs.flatten()
            for i in range(5):
                seed = args.seed + i
                test_info = get_multitask_test_function(seed, n_target=args.n_target, n_source=args.n_source)
                plot_multitask(axs[0][i], axs[1][i], test_info, pfn_info)
            
            plt.legend()
            plt.suptitle(pfn_info[0])
            plt.tight_layout()
            plt.savefig(f"figures/pfn_{pfn_info[0]}_multitask.png")
            plt.close()
        except Exception as e:
            traceback.print_exc()
            print("failed to load", pfn_dir)
            continue
            
        