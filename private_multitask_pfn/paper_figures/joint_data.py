#!/usr/bin/env python3
import gpytorch

from PFNs.pfns.priors import Batch
import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel
import os
import matplotlib.pyplot as plt
import numpy as np
import timeit

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
        
def get_mtgp_batch(n_samples, n_features, n_tasks):
    xs = torch.rand(n_samples, n_features)
    task_id = torch.randint(n_tasks, size=(n_samples,)).unsqueeze(1).long()
    
    rbf = RBFKernel()
    rbf.lengthscale = torch.tensor(0.2)
    covar_x = rbf(xs)
    task_covar_matrix = torch.eye(n_tasks)
    # randomly choose two tasks to have a correlation
    related_tasks = torch.randperm(n_tasks)[:2]
    task_covar_matrix[related_tasks[0], related_tasks[1]] = 0.9
    task_covar_matrix[related_tasks[1], related_tasks[0]] = 0.9
    
    covar_t = task_covar_matrix[task_id, task_id.t()].squeeze()
    covar = covar_x.mul(covar_t)
    ys = MultivariateNormal(torch.zeros(n_samples), covar).sample()

    return xs, task_id, ys


def draw_paired(n, target_corr, samples):
    
    target_train = torch.linspace(0, 0.25, n)
    target_test = torch.linspace(0.25, 1, n)
    
    source1_train = torch.linspace(0.4, 1, n)
    source2_train = torch.linspace(0.0, 0.6, n)
    
    xs = torch.cat([target_train, source1_train, source2_train, target_test], dim=0).view(-1, 1)
    task_id = torch.cat([torch.zeros(n), torch.ones(n), torch.ones(n) * 2, torch.zeros(n)], dim=0).long()
    task_id = task_id.unsqueeze(1)
    
    rbf = RBFKernel()
    rbf.lengthscale = torch.tensor(0.1)
    covar_x = rbf(xs)
    task_covar_matrix = torch.eye(3)
    task_covar_matrix[0, 1] = target_corr
    task_covar_matrix[1, 0] = target_corr
    task_covar_matrix[0, 2] = target_corr
    task_covar_matrix[2, 0] = target_corr
    task_covar_matrix[1, 2] = 1.0
    task_covar_matrix[2, 1] = 1.0
    
    covar_t = task_covar_matrix[task_id, task_id.t()].squeeze()
    covar = covar_x.mul(covar_t)
    # covar = covar_x
    ys = MultivariateNormal(torch.zeros(4 * n), covar).sample(torch.Size([samples]))
    
    # add batch dimension to xs
    return xs.unsqueeze(0).repeat(samples, 1, 1), task_id.unsqueeze(0).repeat(samples, 1, 1), ys


def get_constructed(n, pairs, samples=10):
    source_xs = []
    source_ys = []
    target_train_xs = []
    target_train_ys = []
    target_test_xs = []
    target_test_ys = []
    
    for i in range(pairs):
        if i == 0:
            corr = 0.95
        else:
            corr = 1.0
        xs, task_id, ys = draw_paired(n, corr, samples)
        target_train, source1_train, source2_train, target_test = xs[:, :n], xs[:, n:2 * n], xs[:, 2 * n:3 * n], xs[:, 3 * n:]
        target_y, source1_y, source2_y, target_test_y = ys[:, :n], ys[:, n:2 * n], ys[:, 2 * n:3 * n], ys[:, 3 * n:]
        
        source_xs.extend([source1_train, source2_train])
        source_ys.extend([source1_y, source2_y])
        target_train_xs.append(target_train)
        target_train_ys.append(target_y)
        target_test_xs.append(target_test)
        target_test_ys.append(target_test_y)

    target_train_xs = target_train_xs[0]
    # target_train_ys = torch.stack(target_train_ys, dim=0).sum(dim=0)
    target_train_ys = target_train_ys[0]
    target_train_ids = torch.zeros(samples, n).long()
    target_test_xs = target_test_xs[0]
    # target_test_ys = torch.stack(target_test_ys, dim=0).sum(dim=0)
    target_test_ys = target_test_ys[0]
    target_test_ids = torch.zeros(samples, n).long()
    
    source_xs = torch.cat(source_xs, dim=1)
    source_ys = torch.cat(source_ys, dim=1)
    source_ids = torch.cat([torch.ones(samples, n) * i for i in range(pairs * 2)], dim=1).long() + 1
    
    xs = torch.cat([target_train_xs, source_xs, target_test_xs], dim=1)
    task_id = torch.cat([target_train_ids, source_ids, target_test_ids], dim=1).unsqueeze(-1)
    ys = torch.cat([target_train_ys, source_ys, target_test_ys], dim=1)
    
    return xs, task_id, ys.unsqueeze(-1)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
def get_joint_batch(
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
    xs, task_id, ys = get_constructed(seq_len // 4, 1, batch_size)
    perm = torch.randperm(xs.size(0))
    # randomly permute the data
    xs = xs[perm]
    task_id = task_id[perm]
    ys = ys[perm]
    
    return Batch(
        x=xs.transpose(0, 1),
        y=ys.transpose(0, 1),
        target_y=ys.transpose(0, 1).clone(),
        task_id=task_id.transpose(0, 1),
    )

    

def mtgp_predict(task_id, train_x, train_y, possible_x):
    mtgp = mtgp_fit(task_id.double(), train_x.double(), train_y.double())
    covar_factor = mtgp.task_covar_module.covar_factor.detach()
    v = mtgp.task_covar_module.raw_var.detach()
    cov_matrix = covar_factor @ covar_factor.t() + torch.eye(covar_factor.size(0)) * np.exp(v)
    # to correlation
    std_devs = np.sqrt(np.diagonal(cov_matrix))
    # Outer product of the standard deviations to get normalization factor
    # print(cov_matrix)
    corr_matrix = cov_matrix / (std_devs[:, None] * std_devs[None, :])
    print(corr_matrix)
    possible_x, _ = to_mtgp_format(torch.zeros_like(possible_x[..., 0]), possible_x, torch.zeros_like(possible_x))
    return mtgp.posterior(possible_x)



def joint_visualization():
    


    for seed in range(10, 20):
        torch.manual_seed(seed)

        n = 50
        n_test = n
        pairs = 1
        xs, tasks, ys = get_constructed(n, pairs)
        print(xs.shape, tasks.shape, ys.shape)
        train_xs, train_tasks, train_ys = xs[0, :-n_test], tasks[0, :-n_test].squeeze(), ys[0, :-n_test]
        test_xs, test_tasks, test_ys = xs[0, -n_test:], tasks[0, -n_test:].squeeze(), ys[0, -n_test:]


        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        def plot(ax, pred, title, legend=False):
            if "NUTS" in title:
                pred_mean, pred_std = pred.mixture_mean.detach().squeeze(), pred.mixture_variance.sqrt().detach().squeeze()
            else:
                pred_mean = pred.mean.detach().squeeze()
                pred_std = pred.variance.sqrt().detach().squeeze()
            
            print(title, pred_mean.shape, pred_std.shape, test_ys.shape)
            nll = -torch.distributions.Normal(pred_mean, pred_std).log_prob(test_ys.squeeze()).mean()
            colors = plt.cm.tab10.colors + ("darkgrey",)
            # colors = colors[:1] + colors[2:] + ("darkgrey",)
            for i in range(pairs * 2 + 1):
                if i == 0:
                    label = "Target"
                else:
                    label = f"Auxiliary {i}"
                ax.scatter(train_xs[train_tasks == i], train_ys[train_tasks == i], label=label, color=colors[i], marker="x")
            ax.plot(test_xs.squeeze(), test_ys.squeeze(), color=colors[0], label="True")
            ax.plot(test_xs.squeeze(), pred_mean, color=colors[-1])
            ax.fill_between(test_xs.squeeze(), pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.2, color=colors[-1], label=r"$\mu \pm 2\sigma$")
            ax.axvline(0.6, color="black", linestyle="--", alpha=0.5, label="End of Aux 1")
            if legend:
                ax.legend()
            ax.set_title(title + " NLL: {:.2f}".format(nll.item()))
            
            
        # pfn_dir = "/home/yl9959/mtpfn/wandb_links/%s" % "dutiful-darkness-309"
        # pfn_dir = "/home/yl9959/mtpfn/wandb_links/%s" % "mild-valley-277"
        # pfn_dir = "/home/yl9959/mtpfn/wandb_links/%s" % "major-monkey-325"
        try:
            pfn_dir = "/home/yl9959/mtpfn/ckpt/25-01-15_22-04-28__prior_mtgp__features_1__tasks_3__epochs_50__seqlen_80__attn_standard__task_hier__seed_0__13187" 
            pfn = load_model(pfn_dir)
        except FileNotFoundError:
            pfn_dir = "/home/lily_l/private_multitask_pfn/ckpt/25-01-15_22-04-28__prior_mtgp__features_1__tasks_3__epochs_50__seqlen_80__attn_standard__task_hier__seed_0__13187"
            pfn = load_model(pfn_dir)
        except e:
            raise e
        pred = pfn_predict(pfn, train_tasks, train_xs, train_ys, test_xs)
        plot(axs[0], pred, "MTPFN")
        
        pred = mtgp_predict(train_tasks, train_xs, train_ys, test_xs)
        plot(axs[1], pred, "Joint (ICM)")

        pred = scaml_predict(train_tasks, train_xs, train_ys, test_xs)
        plot(axs[2], pred, "Ensemble (ScaML)")
        
        min_y = np.min([ax.get_ylim()[0] for ax in axs])
        max_y = np.max([ax.get_ylim()[1] for ax in axs])
        for ax in axs:
            ax.set_ylim(min_y, max_y)
        
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pred = mtgp_nuts_predict(train_tasks.to(device), train_xs.to(device), train_ys.to(device), test_xs.to(device))
        # plot(axs[2], pred, "MTGP NUTS")

        # 3 column on top
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
        # plt.tight_layout()
        plt.savefig("joint_data_{}.png".format(seed), bbox_inches="tight")
        plt.savefig("joint_data_{}.pdf".format(seed), bbox_inches="tight")
        
        # break
        
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out!")

def long_running_function():
    signal.signal(signal.SIGALRM, timeout_handler)  # Set the signal handler
    signal.alarm(600)  # Set the alarm for 600 seconds (10 minutes)
    try:
        # Your long-running code here
        while True:
            print("Running...")
            # Simulate work
    except TimeoutException:
        print("Function terminated after 10 minutes.")
    finally:
        signal.alarm(0)  # Disable the alarm

# long_running_function()


def time_up_to(func, number, max_time=600):
    signal.signal(signal.SIGALRM, timeout_handler)  # Set the signal handler
    signal.alarm(max_time)  # Set the alarm for 600 seconds (10 minutes)
    time = None
    try:
        time = timeit.timeit(func, number=number) / number
    except TimeoutException:
        print("Function terminated after 10 minutes.")
    finally:
        signal.alarm(0)  # Disable the alarm
    return time

        
def get_times(n_tasks, n_samples_per_task, pfn, device, benchmark_repeats):
    
    print("n_samples_per_task", n_samples_per_task, "n_tasks", n_tasks, flush=True)

    def gen_data(n_tasks, n_samples_per_task):
        n_samples = n_tasks * n_samples_per_task
        xs = torch.rand(n_samples, 1)
        task_id = torch.randint(n_tasks, size=(n_samples,)).long()#.unsqueeze(0)
        ys = torch.randn(n_samples, 1)#.unsqueeze(0)
        
        return task_id.to(device), xs.to(device), ys.to(device)
    
    def benchmark_mtgp():
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        mtgp_fit(task_id, xs, ys)
        
    def benchmark_scaml():
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        for _ in range(n_tasks):
            gp_fit(task_id, xs, ys)
        
    def benchmark_pfn(pfn):
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        pfn_gaussian_fit(pfn, task_id, xs, ys)
    
    mtgp_time = time_up_to(benchmark_mtgp, number=benchmark_repeats) if n_tasks < 15 else None
    scaml_times = time_up_to(benchmark_scaml, number=benchmark_repeats) if n_tasks < 100 else None
    pfn_times = time_up_to(lambda: benchmark_pfn(pfn), number=benchmark_repeats)
    
    return mtgp_time, scaml_times, pfn_times
        
        
        
def benchmark_time():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_repeats = 5
    
    # pfn_dir = "/home/yl9959/mtpfn/wandb_links/%s" % "dutiful-darkness-309"
    # pfn_dir = "/home/yl9959/mtpfn/ckpt/25-01-15_22-04-28__prior_mtgp__features_1__tasks_3__epochs_50__seqlen_80__attn_standard__task_hier__seed_0__13187"
    pfn_dir = "/home/yl9959/mtpfn/ckpt/25-01-16_11-13-16__prior_mtgp__features_1__tasks_3__epochs_50__seqlen_200__attn_standard__task_hier__seed_0__98742"
    pfn = load_model(pfn_dir)
    
    mtgp_times = []
    scaml_times = []
    pfn_times = []
    
    n_samples_per_task = 50
    n_taskss = torch.logspace(0.5, 4, 20).int()
    for n_tasks in n_taskss:
        mtgp_time, scaml_time, pfn_time = get_times(n_tasks, n_samples_per_task, pfn, device, benchmark_repeats)
        mtgp_times.append(mtgp_time)
        scaml_times.append(scaml_time)
        pfn_times.append(pfn_time)
        
        plt.plot(n_taskss[:len(mtgp_times)], mtgp_times, label="Joint (ICM)")
        plt.plot(n_taskss[:len(mtgp_times)], scaml_times, label="Ensemble (ScaML)")
        plt.plot(n_taskss[:len(mtgp_times)], pfn_times, label="PFN")
        plt.xlabel("Tasks")
        plt.title(f"{n_samples_per_task} Samples per Task")
        
        # save csv
        df = pd.DataFrame({
            "n_tasks": n_taskss[:len(mtgp_times)],
            "mtgp": mtgp_times,
            "scaml": scaml_times,
            "pfn": pfn_times
        })
        df.to_csv("benchmark_time_task.csv")
        
        plt.xscale("log")
        plt.ylabel("Time (s)")
        plt.yscale("log")
        plt.legend()
        plt.savefig("benchmark_time_task.png")
        plt.savefig("benchmark_time_task.pdf")
        plt.clf()
        
    return
        
    
    mtgp_times = []
    scaml_times = []
    pfn_times = []
    
    n_tasks = 5
    n_samples_per_tasks = torch.logspace(1.5, 2.5, 10).int()
    for n_samples_per_task in n_samples_per_tasks:
        mtgp_time, scaml_time, pfn_time = get_times(n_tasks, n_samples_per_task, pfn, device, benchmark_repeats)
        mtgp_times.append(mtgp_time)
        scaml_times.append(scaml_time)
        pfn_times.append(pfn_time)
        
        plt.plot(n_samples_per_tasks[:len(mtgp_times)], mtgp_times, label="Joint (ICM)")
        plt.plot(n_samples_per_tasks[:len(mtgp_times)], scaml_times, label="Ensemble (ScaML)")
        plt.plot(n_samples_per_tasks[:len(mtgp_times)], pfn_times, label="PFN")
        plt.xlabel("Samples per Task")
        plt.title(f"5 Tasks")
        
        # save csv
        df = pd.DataFrame({
            "n_samples_per_task": n_samples_per_tasks[:len(mtgp_times)],
            "mtgp": mtgp_times,
            "scaml": scaml_times,
            "pfn": pfn_times
        })
        df.to_csv("benchmark_time_samples.csv")
        
        plt.xscale("log")
        plt.ylabel("Time (s)")
        plt.yscale("log")
        plt.legend()
        plt.savefig("benchmark_time_samples.png")
        plt.savefig("benchmark_time_samples.pdf")
        plt.clf()
    
        
def get_specific_times(model_name, n_tasks, n_samples_per_task, device, benchmark_repeats):
    
    print(model_name, "n_samples_per_task", n_samples_per_task, "n_tasks", n_tasks, flush=True)

    def gen_data(n_tasks, n_samples_per_task):
        n_samples = n_tasks * n_samples_per_task
        xs = torch.rand(n_samples, 1)
        task_id = torch.randint(n_tasks, size=(n_samples,)).long()#.unsqueeze(0)
        ys = torch.randn(n_samples, 1)#.unsqueeze(0)
        
        return task_id.to(device), xs.to(device), ys.to(device)
    
    def benchmark_mtgp():
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        mtgp_fit(task_id, xs, ys)
        
    def benchmark_scaml():
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        for _ in range(n_tasks):
            gp_fit(task_id, xs, ys)
        
    def benchmark_pfn(pfn):
        task_id, xs, ys = gen_data(n_tasks, n_samples_per_task * n_tasks)
        pfn_gaussian_fit(pfn, task_id, xs, ys)
    
    if model_name == "mtgp":
        benchmark = benchmark_mtgp
    elif model_name == "scaml":
        benchmark = benchmark_scaml
    time = time_up_to(benchmark_mtgp, number=benchmark_repeats)
    
    return time
        
        
def benchmark_time_slurm(model_name, vary_tasks=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_repeats = 20

    result_times = []
    df = pd.DataFrame()
    if vary_tasks:
        n_samples_per_task = 50
        n_taskss = torch.logspace(0.5, 4, 20).int()
        for n_tasks in n_taskss:
            times = get_specific_times(model_name, n_tasks, n_samples_per_task, device, benchmark_repeats)
            result_times.append(times)
            
            df = pd.DataFrame({
                "n_tasks": n_taskss[:len(result_times)],
                "n_samples_per_task": [n_samples_per_task] * len(result_times),
                "model": [model_name] * len(result_times),
                "time": result_times,
            })
            df.to_csv(f"benchmark_time_{model_name}_task.csv", index=False)
            
    else:
        n_tasks = 5
        n_samples_per_tasks = torch.logspace(1.5, 2.5, 10).int()
        
        for n_samples_per_task in n_samples_per_tasks:
            times = get_specific_times(model_name, n_tasks, n_samples_per_task, device, benchmark_repeats)
            result_times.append(times)
            
            df = pd.DataFrame({
                "n_tasks": [n_tasks] * len(result_times),
                "n_samples_per_task": n_samples_per_tasks[:len(result_times)],
                "model": [model_name] * len(result_times),
                "time": result_times,
            })
            df.to_csv(f"benchmark_time_{model_name}_samples.csv", index=False)
        


if __name__ == "__main__":
    from utils import *#mtgp_fit, to_mtgp_format, scaml_predict, mtgp_nuts_predict, pfn_predict, load_model
    import sys
    
    # # benchmark time
    # arg = sys.argv[1]
    # arg_options = [
    #     ("mtgp", True),
    #     ("scaml", True),
    #     ("mtgp", False),
    #     ("scaml", False),
    # ]
    # args = arg_options[int(arg)]
    # benchmark_time_slurm(*args)
    
    # visualization
    joint_visualization()