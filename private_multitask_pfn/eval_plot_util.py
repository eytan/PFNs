import itertools

import multiprocessing as mp

import botorch
import torch
import pickle
from eval_fcnet import (
    get_torch_format_fcnet,
)
from eval_hpobench import (
    get_torch_format_hpobench,
)
import json

from gen_batch import get_mtgp_for_eval
from utils import (
    create_train_test,
    load_model,
    pfn_predict,
    mtgp_predict,
    gp_predict,
    scaml_predict,
    standardize_by_task,
    pfn_get_best,
    mtgp_get_best,
    gp_get_best,
    scaml_get_best,
)
import time
import datetime
import warnings
import os
import hashlib
import argparse
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict


model_to_label = {
    # "different-dust-282": "PFN (0.5 uncorr)",
    # "royal-firebrand-281": "PFN (0.25 uncorr)",
    # "peach-plasma-280": "PFN (1.0 uncorr)",
    # "treasured-lion-279": "PFN (0.0 uncorr)",
    # "exalted-wave-278": "PFN (0.75 uncorr)",
    "wobbly-donkey-484": "wobbly-donkey-484 gaussian 0.1",
    "sparkling-microwave-480": "sparkling-microwave-480 gaussian 0",
    "likely-donkey-479": "likely-donkey-479 gaussian 0.2",
    "fallen-cherry-483": "fallen-cherry-483 bar 0",
    "stilted-night-482": "stilted-night-482 bar 0.1",
    "youthful-wildflower-481": "youthful-wildflower-481 bar 0.2",
}

baselines = ["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]


def get_label(model):
    if model in model_to_label:
        return model_to_label[model]
    return model


def get_labels(models):
    return [get_label(model) for model in models]


def plot_all(exp_dir, n_functions, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    # one plot for boxplots of mse, one plot for boxplots of nll
    
    model_mses = defaultdict(list)
    model_nlls = defaultdict(list)
    
    for function_dir in os.listdir(exp_dir):
        # check if directory
        if not os.path.isdir(os.path.join(exp_dir, function_dir)):
            continue
        with open(os.path.join(exp_dir, function_dir, "results.json"), "r") as f:
            results = json.load(f)
            
        models = list(results.keys())
        for model in models:
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
            
    # plot mse boxplots
    ax = axs[0]
    ax.boxplot([model_mses[model] for model in model_mses], showfliers=False)
    ax.set_xticklabels(get_labels(model_mses.keys()))
    # rotate x labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_title("MSE")
    
    # plot nll boxplots
    ax = axs[1]
    ax.boxplot([model_nlls[model] for model in model_nlls], showfliers=False)
    ax.set_xticklabels(get_labels(model_nlls.keys()))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_title("NLL")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plot.png"))


def plot_function(function_dir, n_trials, title):
    max_scatters = 50
    with open(os.path.join(function_dir, "results.json"), "r") as f:
        results = json.load(f)
        
    models = list(results.keys())
    mse_dict = {}
    nll_dict = {}
        
    fig, axs = plt.subplots(1, len(models) + 2, figsize=(6 * (len(models) + 2), 7))
    axs = axs.flatten()
    for i, model in enumerate(models):
        try:
            mses, nlls = [], []
            for trial in range(n_trials):
                trial_key = "trial_%d" % trial
                mses.append(results[model][trial_key]["mse"])
                nlls.append(results[model][trial_key]["nll"])
            mse_dict[model] = mses
            nll_dict[model] = nlls
            
            ax = axs[i]
            # visualize predictions for one trial
            mean = torch.tensor(results[model]["trial_0"]["mean"]).squeeze()
            std = 2 * torch.tensor(results[model]["trial_0"]["std"]).squeeze()
            true = torch.tensor(results[model]["trial_0"]["true"]).squeeze()
            
            ax.scatter(true[:max_scatters], mean[:max_scatters])
            ax.errorbar(true[:max_scatters], mean[:max_scatters], yerr=std[:max_scatters], fmt="o")
            ax.plot([true[:max_scatters].min(), true[:max_scatters].max()], [true[:max_scatters].min(), true[:max_scatters].max()], "k--")
            ax.set_title(get_label(model) + " Predictions")
            
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            traceback.print_exc()
            continue
        
    all_mses = [result for model in mse_dict for result in mse_dict[model]]
    all_nlls = [result for model in nll_dict for result in nll_dict[model]]
    all_mses = torch.tensor(all_mses)
    all_nlls = torch.tensor(all_nlls)
    
    
    lower_mse, upper_mse = torch.quantile(all_mses, torch.tensor([0.2, 0.8]))
    diff = upper_mse - lower_mse
    lower_mse = (lower_mse - 3 * diff)
    upper_mse = (upper_mse + 3 * diff)
    all_mses = all_mses[all_mses < upper_mse]
    all_mses = all_mses[all_mses > lower_mse]
    lower_mse, upper_mse = all_mses.min().item(), all_mses.max().item()
    
    lower_nll, upper_nll = torch.quantile(all_nlls, torch.tensor([0.2, 0.8]))
    
    ax = axs[-2]
    for model in mse_dict:
        ax.hist(mse_dict[model], bins=8, alpha=0.5, label=get_label(model), range=(lower_mse, upper_mse), density=True)
    ax.set_title("MSE")
    ax.legend()
    
    ax = axs[-1]
    for model in nll_dict:
        ax.hist(nll_dict[model], bins=8, alpha=0.5, label=get_label(model), range=(lower_nll.item(), upper_nll.item()), density=True)
    ax.set_title("NLL")
    ax.legend()
                
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(function_dir, "plot.png"))


def plot_bo_function(function_dir, title):
    plt.figure(figsize=(8, 6))
        
    with open(os.path.join(function_dir, "bo_results.json"), "r") as f:
        bo_results = json.load(f)
    with open(os.path.join(function_dir, "info.json"), "r") as f:
        info = json.load(f)
        
        for model in bo_results:
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
                std = model_results.std(0) / len(model_results)# ** 0.5
                lower = mean - std
                upper = mean + std
                # mean = model_results.mean(0)
                # lower = model_results.quantile(0.25, 0)
                # upper = model_results.quantile(0.75, 0)
                linestyle = "--" if model in baselines else "-"
                plt.plot(range(len(mean)), mean, label=get_label(model), linestyle=linestyle)
                plt.fill_between(range(len(mean)), lower, upper, alpha=0.2)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(e)
                # import pdb; pdb.set_trace()
                # raise e
                continue
                    
    plt.legend()
    plt.xlabel("BO Iterations")
    plt.ylabel("Normalized Regret")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(function_dir, "bo_loop.png"))


def plot_bo_all(exp_dir, models, title):
    plt.figure(figsize=(8, 6))
    
    if models is None:
        models = []
        
    for model in models + ["random", "scaml", "mtgp", "gp", "lmc", "mtgp_nuts"]:
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
                if not torch.isnan(function_results.mean(0)).any():
                    model_results.append(function_results.mean(0))
            except KeyboardInterrupt as e:
                raise e
            except FileNotFoundError as e:
                continue
            except Exception as e:
                # print(e)
                continue
            
        if len(model_results) == 0:
            continue
        model_results = torch.stack(model_results)
            
        # mean = model_results.mean(0)
        # lower = model_results.quantile(0.25, 0)
        # upper = model_results.quantile(0.75, 0)
        
        mean = model_results.nanmean(0)
        std = model_results.std(0) / len(model_results)# ** 0.5
        lower = mean - std
        upper = mean + std
        linestyle = "--" if model in baselines else "-"
        plt.plot(range(len(mean)), mean, label=get_label(model), linestyle=linestyle)
        plt.fill_between(range(len(mean)), lower, upper, alpha=0.2)

                    
    plt.legend()
    plt.xlabel("BO Iterations")
    plt.ylabel("Normalized Regret")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "bo_loop.png"))



