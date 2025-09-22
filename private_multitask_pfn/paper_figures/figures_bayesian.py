from PFNs.pfns.utils import get_restarting_cosine_schedule_with_warmup
import torch
import json


import pickle
import pdb
import gpytorch
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.exceptions import ModelFittingError
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
import botorch
import einops

from tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.model.loading import get_encoder, _preprocess_config, get_y_encoder

from tabpfn import TabPFNClassifier, TabPFNRegressor
import matplotlib.pyplot as plt
import os

import traceback
from collections import defaultdict


colors = {
    "MTPFN": "deepskyblue",
    "ScaML": "#E78C35",
    "ICM": "#DA2222",
    "ICM (NUTS)": "C2",
    "GP": "#95211B",
}

baselines = ["scaml", "mtgp", "gp", "lmc", "mtgp_nuts"] #+ ["random"] 

# model_to_label = {
#     "different-dust-282": "PFN (0.5 uncorr)",
#     "royal-firebrand-281": "PFN (0.25 uncorr)",
#     "peach-plasma-280": "PFN (1.0 uncorr)",
#     "treasured-lion-279": "PFN (0.0 uncorr)",
#     "exalted-wave-278": "PFN (0.75 uncorr)",
# }

# baselines = ["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]


def get_label(model):
    if model in model_to_label:
        return model_to_label[model]
    return model


def get_labels(models):
    return [get_label(model) for model in models]


def get_color(model):
    return colors[get_label(model)]





def plot_bo_all(exp_dir, models, title, path):
    plt.figure(figsize=(8, 6))
    
    if models is None:
        models = []
        
    # for model in models + ["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]:
    for model in baselines + models:
        model_results = []
        for function_dir in os.listdir(exp_dir):
            try:
                function_results = []
                # check if directory
                if not os.path.isdir(os.path.join(exp_dir, function_dir)):
                    continue
                with open(os.path.join(exp_dir, function_dir, "bo_results.json"), "r") as f:
                    bo_results = json.load(f)
                with open(os.path.join(exp_dir, function_dir, "info.json"), "r") as f:
                    info = json.load(f)
                
                
                for trial in bo_results[model]:
                    best_init = info[trial]["best_init"]
                    best_possible = info[trial]["best_possible"]
                    
                    results = bo_results[model][trial]
                        
                    train_y = torch.tensor([[best_init]] + results["train_y"]).squeeze()
                    cummax = torch.tensor(train_y).cummax(0).values.squeeze()
                    # average normalized regret
                    regret = (cummax - best_possible) / (best_init - best_possible)
                    
                    function_results.append(regret)
                
                function_results = torch.stack(function_results)
                model_results.append(function_results.mean(0))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                continue
            
        if len(model_results) == 0:
            continue
        model_results = torch.stack(model_results)
            
        # import pdb; pdb.set_trace()
        # mean = model_results.median(0).values
        # lower = model_results.quantile(0.25, 0)
        # upper = model_results.quantile(0.75, 0)
        mean = model_results.mean(0)
        std = model_results.std(0) / 2 / len(model_results)# ** 0.5
        lower = mean - std
        upper = mean + std
        linestyle = "--" if model in baselines else "-"
        # linestyle = "-" if model in baselines else "--"
        # linestyle = "-"
        markevery = slice(0, 30, 3) if model in baselines else slice(2, 30, 3)
        # linestyle 
        plt.plot(range(len(mean)), mean, label=get_label(model), linestyle=linestyle, color=get_color(model), linewidth=1)#, marker=marker[get_label(model)], markersize=8, markevery=markevery)#, markeredgecolor="black")
        plt.fill_between(range(len(mean)), lower, upper, alpha=0.15, color=get_color(model))

                    
    plt.legend()
    plt.xlabel("BO Iterations")
    plt.ylabel("Normalized Regret")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    
    
    

def plot_bo_ax(ax, exp_dir, models, title, path=None):    
    if models is None:
        models = []
        
    # for model in models + ["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]:
    for model in baselines + models:
        model_results = []
        for function_dir in os.listdir(exp_dir):
            try:
                function_results = []
                # check if directory
                if not os.path.isdir(os.path.join(exp_dir, function_dir)):
                    continue
                with open(os.path.join(exp_dir, function_dir, "bo_results.json"), "r") as f:
                    bo_results = json.load(f)
                with open(os.path.join(exp_dir, function_dir, "info.json"), "r") as f:
                    info = json.load(f)
                
                
                for trial in bo_results[model]:
                    best_init = info[trial]["best_init"]
                    best_possible = info[trial]["best_possible"]
                    
                    results = bo_results[model][trial]
                        
                    train_y = torch.tensor([[best_init]] + results["train_y"]).squeeze()
                    cummax = torch.tensor(train_y).cummax(0).values.squeeze()
                    # average normalized regret
                    regret = (cummax - best_possible) / (best_init - best_possible)
                    
                    function_results.append(regret)
                
                function_results = torch.stack(function_results)
                if torch.isnan(function_results).any():
                    continue
                model_results.append(function_results.mean(0))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                continue
            
        if len(model_results) == 0:
            continue
        model_results = torch.stack(model_results)
            
        # import pdb; pdb.set_trace()
        # mean = model_results.median(0).values
        # lower = model_results.quantile(0.25, 0)
        # upper = model_results.quantile(0.75, 0)
        mean = model_results.mean(0)
        std = model_results.std(0) / 2 / len(model_results) #0.5
        lower = mean - std
        upper = mean + std
        linestyle = "--" if model in baselines else "-"
        # linestyle = "-" if model in baselines else "--"
        # linestyle = "-"
        # linestyle 
        xs = torch.linspace(0, 40, len(mean))
        ax.plot(xs, mean, label=get_label(model), linestyle=linestyle, color=get_color(model), linewidth=2)#, marker=marker[get_label(model)], markersize=8, markevery=markevery)#, markeredgecolor="black")
        ax.fill_between(xs, lower, upper, alpha=0.15, color=get_color(model))

    ax.set_title(title)
    ax.set_ylabel("Normalized Regret")
    ax.set_xlabel("BO Iterations")
    # ax.set_ylim(0.05, 0.7)
    # ax.set_yscale("log")
    # plt.legend()
    # plt.xlabel("BO Iterations")
    # plt.ylabel("Normalized Regret")
    # plt.title(title)
    # plt.tight_layout()
    # plt.savefig(path)
    
    
    
    

def plot_all(exp_dir, models, title, path):
    plt.figure(figsize=(8, 6))
    # one plot for boxplots of mse, one plot for boxplots of nll
    
    model_mses = defaultdict(list)
    model_nlls = defaultdict(list)
    
    all_models = models + baselines #["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]

    for model_order in all_models:
        for function_dir in os.listdir(exp_dir):
            # check if directory
            if not os.path.isdir(os.path.join(exp_dir, function_dir)):
                continue
            with open(os.path.join(exp_dir, function_dir, "results.json"), "r") as f:
                results = json.load(f)
                
            models = list(results.keys())
            for model in models:
                if model != model_order:
                    continue
                mses, nlls = [], []
                try:
                    for trial_key in results[model]:
                        mses.append(results[model][trial_key]["mse"])
                        nlls.append(results[model][trial_key]["nll"])
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    traceback.print_exc()
                    continue
                model_mses[model].extend(mses)
                model_nlls[model].extend(nlls)
    
    # plot nll boxplots
    plt.boxplot([model_nlls[model] for model in model_nlls], showfliers=True)
    plt.gca().set_xticklabels(get_labels(model_nlls.keys()))
    # for tick in plt.gca().get_xticklabels():
    #     tick.set_rotation(45)
    # plt.title("NLL")
    plt.title(title)
    plt.ylabel("NLL")
    # plt.yscale("log")
    
    # plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    
    # plot mse boxplots
    plt.boxplot([model_mses[model] for model in model_mses], showfliers=True)
    plt.gca().set_xticklabels(get_labels(model_mses.keys()))
    # for tick in plt.gca().get_xticklabels():
    #     tick.set_rotation(45)
    # plt.title("MSE")
    plt.title(title)
    plt.ylabel("MSE")
    # plt.yscale("log")
    
    # plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_mse.png"))
    plt.clf()
    

def plot_bo_function_ax(ax, function_dir, models, title):
    with open(os.path.join(function_dir, "bo_results.json"), "r") as f:
        bo_results = json.load(f)
    with open(os.path.join(function_dir, "info.json"), "r") as f:
        info = json.load(f)
        
    for model in baselines + models:
        for model_try in bo_results:
            if model_try != model:
                continue
            model_results = []
            try:
                for trial in bo_results[model]:
                    best_init = info[trial]["best_init"]
                    best_possible = info[trial]["best_possible"]
                    
                    results = bo_results[model][trial]
                        
                    train_y = torch.tensor([[best_init]] + results["train_y"]).squeeze()
                    cummax = torch.tensor(train_y).cummax(0).values.squeeze()
                    # average normalized regret
                    regret = (cummax - best_possible) / (best_init - best_possible)
                    
                    model_results.append(regret)
                
                # remove shorter trials
                max_length = max([len(result) for result in model_results])
                model_results = [result for result in model_results if len(result) == max_length]
                
                model_results = torch.stack(model_results)
                
                mean = model_results.nanmean(0)
                std = model_results.std(0) /  len(model_results) #** 0.5
                lower = mean - std
                upper = mean + std
                # mean = model_results.mean(0)
                # lower = model_results.quantile(0.25, 0)
                # upper = model_results.quantile(0.75, 0)
                linestyle = "--" if model in baselines else "-"
                
                xs = torch.linspace(0, 40, len(mean))
                ax.plot(xs , mean, label=get_label(model), linestyle=linestyle, color=get_color(model), linewidth=2)#, marker=marker[get_label(model)], markersize=8, markevery=markevery)#, markeredgecolor="black")
                ax.fill_between(xs, lower, upper, alpha=0.15, color=get_color(model))
                # plt.plot(range(len(mean)), mean, label=get_label(model), linestyle=linestyle)
                # plt.fill_between(range(len(mean)), lower, upper, alpha=0.2)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(e)
                # import pdb; pdb.set_trace()
                # raise e
                continue
                    
    # plt.legend()
    ax.set_xlabel("BO Iterations")
    ax.set_ylabel("Normalized Regret")
    ax.set_title(title)
    # plt.title(title)
    # plt.tight_layout()
    # plt.savefig(path)



# models = ["wobbly-donkey-484", "fallen-cherry-483", "stilted-night-482", "youthful-wildflower-481", "sparkling-microwave-480", "likely-donkey-479"]
# plot_bo_all("/home/yl9959/mtpfn/eval_plot/25-01-28_11-31-32__synthetic__trials_5__seed_0__num_samples/features_3__tasks_4__lengthscale_None__task_corr_0_8__uncorr_tasks_0__n_target_2__n_source_20__seed_0", models, "All")

# fig, axs = plt.subplots(1, 5, figsize=(12, 2.5))
plt.figure(figsize=(6, 4))
ax = plt.gca()

# # add grid
# for ax in axs:
#     # y grid only
#     ax.yaxis.grid(True,  linestyle='-', linewidth=0.5, alpha=0.5)

dir = "/home/lily_l/private_multitask_pfn/eval_plot/25-01-30_23-38-18__synthetic__trials_3__seed_0__nuts_1/features_3__tasks_4__lengthscale_0_2__task_corr_0_6__uncorr_tasks_0__n_target_2__n_source_20__seed_0/function_0"
models = ["stilted-night-482"]
model_to_label = {
    "stilted-night-482": "MTPFN",
    "mtgp": "ICM",
    "scaml": "ICM (NUTS)",
    "gp": "GP",
    "mtgp_nuts": "ScaML",
}
title = ""
plot_bo_function_ax(ax, dir, models, title)

plt.legend()
plt.savefig("full.pdf", bbox_inches='tight')
lily


dir = "/home/lily_l/private_multitask_pfn/eval_plot/25-01-30_16-20-48__hpobench__trials_3__seed_0__xgb/n_target_5__n_source_20"
model_to_label = {
    "stilted-night-482": "MTPFN",
    "mtgp": "ICM",
    "gp": "GP",
    "scaml": "ScaML",
}
models = ["stilted-night-482"]
title = "XGB"
plot_bo_ax(axs[1], dir, models, title)



dir = "/home/lily_l/private_multitask_pfn/eval_plot/25-01-30_16-20-47__hpobench__trials_3__seed_0__rf/n_target_2__n_source_20"
model_to_label = {
    "stilted-night-482": "MTPFN",
    "mtgp": "ICM",
    "gp": "GP",
    "scaml": "ScaML",
}
models = ["stilted-night-482"]
title = "RF"
plot_bo_ax(axs[2], dir, models, title)

dir = "/home/lily_l/private_multitask_pfn/eval_plot/25-01-30_16-20-47__hpobench__trials_3__seed_0__lr/n_target_2__n_source_20"
model_to_label = {
    "fallen-cherry-483": "MTPFN",
    "mtgp": "ICM",
    "gp": "GP",
    "scaml": "ScaML",
}
models = ["fallen-cherry-483"]
title = "LR"
plot_bo_ax(axs[3], dir, models, title)

dir = "/home/lily_l/private_multitask_pfn/eval_plot/25-01-30_16-20-48__hpobench__trials_3__seed_0__nn/n_target_2__n_source_20"
model_to_label = {
    "stilted-night-482": "MTPFN",
    "mtgp": "ICM",
    "gp": "GP",
    "scaml": "ScaML",
}
models = ["stilted-night-482"]
title = "NN"
plot_bo_ax(axs[4], dir, models, title)

handles, labels = axs[0].get_legend_handles_labels()
# legend above plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, handles=handles, labels=labels)
# plt.legend()
plt.tight_layout()
plt.savefig("hpobench.pdf", bbox_inches='tight')
