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
from utils import *
import datetime
import warnings
import os
import argparse
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict
from eval_plot_util import plot_function, plot_all, plot_bo_function, plot_bo_all

warnings.filterwarnings("ignore")

FUNCTION_SEED_OFFSET = 10
TRIAL_SEED_OFFSET = 1000

def model_predict(
    model_dir,
    model_name,
    train_id,
    train_x,
    train_y,
    test_target_x,
    test_target_y,
    standardize,
    trial_seed,
    rerun,
    model=None,
):
    os.makedirs(model_dir, exist_ok=True)
    
    # check if results exist
    try:
        if rerun:
            raise FileNotFoundError
        # if model_name == "mtgp_nuts":# or model_name == "gp":
        #     raise FileNotFoundError
        with open(os.path.join(model_dir, "results.json"), "r") as f:
            results = json.load(f)
            trial_seed_key = str(trial_seed)
            if trial_seed_key in results:
                return results[trial_seed_key]
    except FileNotFoundError as e:
        results = {}
    except Exception as e:
        traceback.print_exc()
        results = {}

    print("\tPredicting", model_name)
    with botorch.manual_seed(trial_seed):
        try:
            # standardize output to mean 0 std 1
            if standardize:
                assert (0.0 <= train_x).all() and (train_x <= 1.0).all()
                processed_y, processed_test_y = standardize_by_task(train_y, train_id, test_target_y)
            else:
                processed_y = train_y
            assert not processed_y.isnan().any()

            args = (
                train_id,
                train_x,
                processed_y,
                test_target_x,
            )
            if "PFN" in model_name:
                outputs = pfn_predict(model.to(train_x.device), *args)
            elif model_name == "mtgp":
                outputs = mtgp_predict(*args)
            elif model_name == "gp":
                outputs = gp_predict(*args)
            elif model_name == "scaml":
                outputs = scaml_predict(*args)
            elif model_name == "lmc":
                outputs = lmc_predict(*args)
            elif model_name == "mtgp_nuts":
                outputs = mtgp_nuts_predict(*args)
            else:
                raise ValueError(
                    "model_type must be one of ['pfn', 'mtgp', 'gp', 'scaml', 'lmc']"
                )
                
            if model_name == "mtgp_nuts":
                mean, std = outputs.mixture_mean, outputs.mixture_variance.sqrt()
            else:
                mean, std = outputs.mean.squeeze(), outputs.variance.sqrt().squeeze()
            processed_test_y = processed_test_y.squeeze()
            mse = ((mean - test_target_y) ** 2).mean().item()
            nll = -torch.distributions.Normal(mean, std).log_prob(processed_test_y).mean().item()
                
            # save predictions
            results[trial_seed] = {
                "mse": mse,
                "nll": nll,
                "mean": mean.detach().squeeze().tolist(),
                "std": std.detach().squeeze().tolist(),
                "true": processed_test_y.squeeze().tolist(),
            }
            
            del outputs
            
            # print(model_type, results[trial_seed]["mse"], results[trial_seed]["nll"])
                
            with open(os.path.join(model_dir, "results.json"), "w") as f:
                json.dump(results, f)

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # raise e
            # print(e)
            traceback.print_exc()
            print(model_name, "failed")
            return None
        
    return results[trial_seed]



def bo_loop(
    model_dir,
    model_name,
    init_id,
    init_x,
    init_y,
    possible_id,
    possible_x,
    possible_y,
    iters,
    standardize,
    trial_seed,
    rerun,
    model=None
):
    os.makedirs(model_dir, exist_ok=True)
    start_iter = 1
    original_len = len(init_x)
    
    # check if results exist
    try:
        if rerun:
            raise FileNotFoundError
        # if model_name == "gp":
        #     raise FileNotFoundError
        # read length of results
        with open(os.path.join(model_dir, "bo_results.json"), "r") as f:
            results = json.load(f)
            # check if results are complete
            if len(results["train_y"]) < iters:
                print("\tResults are incomplete for %s" % model_dir)
            
            if results["completed_iters"] >= iters:
                return {
                    "iters": iters,
                    "train_y": results["train_y"][:iters],
                    "train_x": results["train_x"][:iters],
                    "task_id": results["task_id"][:iters],
                }
            else:
                # continue from last checkpoint
                half_id = torch.tensor(results["task_id"], device=init_id.device)
                half_x = torch.tensor(results["train_x"], device=init_x.device)
                half_y = torch.tensor(results["train_y"], device=init_y.device)
                trial_seed = results["rng_seed"]
                
                init_id = torch.cat((init_id, half_id), 0)
                init_x = torch.cat((init_x, half_x), 0)
                init_y = torch.cat((init_y, half_y), 0)
                start_iter = results["completed_iters"] + 1
    except FileNotFoundError as e:
        results = {}
    except Exception as e:
        traceback.print_exc()
        results = {}
                
    task_id, train_x, train_y = init_id, init_x, init_y
    print("\tBO Loop for", model_name)
    
    if model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        task_id = task_id.to(device)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        possible_id = possible_id.to(device)
        possible_x = possible_x.to(device)
        possible_y = possible_y.to(device)

    with botorch.manual_seed(trial_seed):
        try:
            for i in range(start_iter, iters + 1):
                # print(model_type, "epoch", i)
                # standardize output to mean 0 std 1
                if standardize:
                    assert (0.0 <= train_x).all() and (train_x <= 1.0).all()
                    processed_y = standardize_by_task(train_y, task_id)
                else:
                    processed_y = train_y
                assert not processed_y.isnan().any()

                args = (
                    task_id,
                    train_x,
                    processed_y,
                    possible_id,
                    possible_x,
                    possible_y,
                )
                if "PFN" in model_name:
                    best_x, best_y = pfn_get_best(model, *args)
                elif model_name == "mtgp":
                    best_x, best_y = mtgp_get_best(*args)
                elif model_name == "gp":
                    best_x, best_y = gp_get_best(*args)
                elif model_name == "scaml":
                    best_x, best_y = scaml_get_best(*args)
                elif model_name == "random":
                    random_index = torch.randint(0, possible_x.shape[0], ())
                    best_x, best_y = possible_x[random_index], possible_y[random_index]
                elif model_name == "lmc":
                    best_x, best_y = lmc_get_best(*args)
                elif model_name == "mtgp_nuts":
                    best_x, best_y = mtgp_nuts_get_best(*args)
                else:
                    raise ValueError(
                        "model_type must be one of ['pfn', 'mtgp', 'gp', 'scaml', 'random', 'lmc']"
                    )

                task_id = torch.cat(
                    (task_id, torch.tensor([0.0], device=task_id.device)), 0
                )
                train_x = torch.cat((train_x, best_x.unsqueeze(0)), 0)
                train_y = torch.cat((train_y, best_y.unsqueeze(0)), 0)
                
                # save results
                seed = torch.seed()
                results = {
                    "completed_iters": i,
                    "train_y": train_y[original_len:].cpu().numpy().tolist(),
                    "train_x": train_x[original_len:].cpu().numpy().tolist(),
                    "task_id": task_id[original_len:].cpu().numpy().tolist(),
                    "rng_seed": seed,
                }
                
                with open(os.path.join(model_dir, "bo_results.json"), "w") as f:
                    json.dump(results, f)

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(model_name, "failed")
            return None
    
    return results
                

def eval_metrics(
    n_trials,
    bo_iters,
    seed,
    baselines,
    pfns,
    data,
    n_target,
    n_source,
    exp_dir,
    cache_dir,
    rerun,
    standardize=True,
):
    # pfns = []
    # if pfn_ids is not None:
    #     for pfn_id in pfn_ids:
    #         try:
    #             pfn_dir = "/home/yl9959/mtpfn/wandb_links/%s" % pfn_id
    #             model = load_model(pfn_dir)
    #             pfns.append((pfn_id, model))
    #         except Exception as e:
    #             traceback.print_exc()
    #             print("failed to load", pfn_id)
    #             continue
        
    xs, ys, sources_xs, sources_ys = data
    
    results = defaultdict(dict)
    bo_results = defaultdict(dict)
    info = defaultdict(dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n!!!!!!", cache_dir, "!!!!!!", flush=True)
    for trial in range(n_trials):
        print("Running trial", trial)
        trial_key = "trial_%d" % trial
        trial_seed = seed + trial * 1000
        
        # generate data
        with botorch.manual_seed(trial_seed):
            try:
                trial_data = torch.load(os.path.join(cache_dir, "data", "trial_%d.pt" % trial), map_location=torch.device('cpu'))
            except FileNotFoundError as e:
                trial_data = create_train_test(
                    xs, ys, sources_xs, sources_ys, n_target, n_source
                )
                os.makedirs(os.path.join(cache_dir, "data"), exist_ok=True)
                torch.save(trial_data, os.path.join(cache_dir, "data", "trial_%d.pt" % trial))
                
            # trial_data = [tensor.to(device) for tensor in trial_data]
            train_id, train_x, train_y, test_id, test_x, test_y = trial_data
            
            args = (
                train_id,
                train_x,
                train_y,
                # test_id,
                test_x,
                test_y,
                standardize,
                trial_seed,
                rerun,
            )

            # for model_type in ["scaml", "mtgp", "gp", "random"]:#, "lmc"]:#, "mtgp_nuts"]:
            for model_name in baselines:
                model_dir = os.path.join(cache_dir, model_name)
                if model_name != "random":
                    results[model_name][trial_key] = model_predict(model_dir, model_name, *args)
                
                bo_model_dir = os.path.join(cache_dir, trial_key, model_name)
                bo_results[model_name][trial_key] = bo_loop(
                    bo_model_dir,
                    model_name,
                    train_id,
                    train_x,
                    train_y,
                    test_id,
                    test_x,
                    test_y,
                    bo_iters,
                    standardize,
                    trial_seed,
                    rerun,
                )
            for pfn_name, pfn in pfns:
                model_dir = os.path.join(cache_dir, pfn_name)
                results[pfn_name][trial_key] = model_predict(model_dir, "PFN: " + pfn_name, *args, model=pfn)
                # os.symlink(model_dir, os.path.join(exp_dir, pfn_name))
                
                bo_model_dir = os.path.join(cache_dir, trial_key, pfn_name)
                bo_results[pfn_name][trial_key] = bo_loop(
                    bo_model_dir,
                    "PFN: " + pfn_name,
                    train_id,
                    train_x,
                    train_y,
                    test_id,
                    test_x,
                    test_y,
                    bo_iters,
                    standardize,
                    trial_seed,
                    rerun,
                    model=pfn,
                )
            
            info[trial_key]["best_possible"] = test_y.max().item()
            info[trial_key]["best_init"] = train_y[train_id == 0].max().item()
            info[trial_key]["init_id"] = train_id.cpu().numpy().tolist()
            info[trial_key]["init_x"] = train_x.cpu().numpy().tolist()
            info[trial_key]["init_y"] = train_y.cpu().numpy().tolist()

                
    # save results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(exp_dir, "bo_results.json"), "w") as f:
        json.dump(bo_results, f)
    with open(os.path.join(exp_dir, "info.json"), "w") as f:
        json.dump(info, f)
                
    return results
                   

def get_eval_suite(eval_type):
    if eval_type == "num_samples":
        # Test impact of number of targets and sources
        # Varying correlations between tasks
        # Varying lengthscales
        # NUM_FEATURES = [3]
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [0.8]
        LENGTHSCALE_OPTIONS = [None]
        NUM_TASKS_OPTIONS = [4]
        # NUM_TARGETS = [2, 5, 10, 20]
        # NUM_SOURCES = [10, 20, 50, 100]
        NUM_TARGETS = [2, 4, 8, 16, 32, 64]
        NUM_SOURCES = [20, 50]
        NUM_SAMPLES = []
        for target in NUM_TARGETS:
            for source in NUM_SOURCES:
                if target <= source:
                    NUM_SAMPLES.append((target, source))

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )

    elif eval_type == "corr":
        # Test impact of number of targets and sources
        # Varying correlations between tasks
        # Varying lengthscales
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [0.3, 0.6, 0.9]
        LENGTHSCALE_OPTIONS = [None, 0.2]
        NUM_TASKS_OPTIONS = [4]
        NUM_SAMPLES = [(2, 20), (5, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )

    elif eval_type == "uncorr":
        # Test number of uncorrelated tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0, 1, 2, 3]
        CORR_INIT_OPTIONS = [None, 0.6, 0.8]
        LENGTHSCALE_OPTIONS = [None, 0.1, 0.2]
        NUM_TASKS_OPTIONS = [4]
        # NUM_SAMPLES = [(2, 20), (5, 20), (10, 20)]
        NUM_SAMPLES = [(2, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )

    elif eval_type == "num_tasks":
        # Test number of tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0, 1]
        CORR_INIT_OPTIONS = [None]
        LENGTHSCALE_OPTIONS = [None]
        NUM_TASKS_OPTIONS = [2, 3, 4, 5, 6]
        NUM_SAMPLES = [(2, 20), (5, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        
    elif eval_type == "one_task":
        # Test number of tasks
        NUM_FEATURES = [1]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [None]
        LENGTHSCALE_OPTIONS = [None, 0.2]
        NUM_TASKS_OPTIONS = [1]
        NUM_SAMPLES = [(4, 20), (8, 20), (16, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        
    elif eval_type == "uncorr_final":
        # Test number of uncorrelated tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0, 1, 2, 3]
        CORR_INIT_OPTIONS = [0.8]
        LENGTHSCALE_OPTIONS = [None]
        NUM_TASKS_OPTIONS = [4]
        NUM_SAMPLES = [(2, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        
    elif eval_type == "nuts_0":
        # Test number of uncorrelated tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [0.3]
        LENGTHSCALE_OPTIONS = [0.2, None]
        NUM_TASKS_OPTIONS = [4]
        NUM_SAMPLES = [(2, 20), (5, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        
    elif eval_type == "nuts_1":
        # Test number of uncorrelated tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [0.6]
        LENGTHSCALE_OPTIONS = [0.2, None]
        NUM_TASKS_OPTIONS = [4]
        NUM_SAMPLES = [(2, 20), (5, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        
    elif eval_type == "nuts_2":
        # Test number of uncorrelated tasks
        NUM_FEATURES = [3]
        NUM_UNCORR_TASKS = [0]
        CORR_INIT_OPTIONS = [0.9]
        LENGTHSCALE_OPTIONS = [0.2, None]
        NUM_TASKS_OPTIONS = [4]
        NUM_SAMPLES = [(2, 20), (5, 20)]

        return list(
            itertools.product(
                NUM_FEATURES,
                NUM_TASKS_OPTIONS,
                LENGTHSCALE_OPTIONS,
                CORR_INIT_OPTIONS,
                NUM_UNCORR_TASKS,
                NUM_SAMPLES,
            )
        )
        


def run_hpobench(
    n_trials: int,
    bo_iters: int,
    seed: int,
    baselines: list,
    pfns: list,
    hpobench_task: str,
    hpobench_n_tasks: int,
    hpobench_n_features: int,
    n_functions: int,
    exp_dir: str,
    cache_dir: str,
    rerun: bool,
    standardize=True,
    **kwargs,
):
    num_features = hpobench_n_features
    top_level_cache_dir = os.path.join(cache_dir, "hpobench")

    torch.manual_seed(seed)
    
    try:
        problem_dfs = pickle.load(open(f"/scratch/yl9959/mtpfn/datasets/hpobench_{hpobench_task}.pkl", "rb"))
    except FileNotFoundError as e:
        problem_dfs = pickle.load(open(f"/home/lily_l/private_multitask_pfn/datasets/hpobench_{hpobench_task}.pkl", "rb"))
    except Exception as e:
        raise e
    possible_ids = list(problem_dfs.keys())
    ids = []
    for _ in range(n_functions):
        id_indices = torch.randperm(len(possible_ids))[:hpobench_n_tasks]
        function_ids = [possible_ids[i] for i in id_indices]
        ids.append((function_ids[0], tuple(function_ids[1:])))

    num_samples = [
        (2, 20),
        (5, 20),
        (20, 20),
        (2, 100),
        (20, 100),
    ]
    for n_target, n_source in num_samples:
        for function, (target_id, source_ids) in enumerate(ids):
            source_ids_str = "_".join([str(id) for id in source_ids])
            exp_exp_name = "n_target_%d__n_source_%d" % (n_target, n_source)
            cache_experiment_name = "%s_features_%d_n_target_%d__n_source_%d__target_%d__source_%s" % (hpobench_task, num_features, n_target, n_source, target_id, source_ids_str)
            
            function_seed = seed + function * FUNCTION_SEED_OFFSET
            
            target_df = problem_dfs[target_id]
            source_dfs = [problem_dfs[source_id] for source_id in source_ids]

            data = get_torch_format_hpobench(
                target_df, source_dfs, num_features, hpobench_task
            )
            
            exp_function_dir = os.path.join(exp_dir, exp_exp_name, "function_%d" % function)
            os.makedirs(exp_function_dir, exist_ok=True)
            
            cache_function_dir = os.path.join(top_level_cache_dir, cache_experiment_name)
            os.makedirs(cache_function_dir, exist_ok=True)
                    
            with botorch.manual_seed(seed):
                eval_metrics(
                    n_trials=n_trials,
                    bo_iters=bo_iters,
                    seed=function_seed,
                    baselines=baselines,
                    pfns=pfns,
                    data=data,
                    n_target=n_target,
                    n_source=n_source,
                    exp_dir=exp_function_dir,
                    cache_dir=cache_function_dir,
                    standardize=standardize,
                    rerun=rerun,
                )
                
            plot_function(exp_function_dir, n_trials, "HPOBench %s: %s\nDataset %d (%d Trials)" % (hpobench_task, exp_exp_name, function + 1, n_trials))
            plot_bo_function(exp_function_dir, "HPOBench %s: %s\nDataset %d (%d Trials)" % (hpobench_task, exp_exp_name, function + 1, n_trials))
            
        pfn_ids = [pfn_id for pfn_id, _ in pfns]
        plot_all(os.path.join(exp_dir, exp_exp_name), n_functions, "HPOBench %s: %s\n Summary over %d datasets and %d trials" % (hpobench_task, exp_exp_name, n_functions, n_trials))
        plot_bo_all(os.path.join(exp_dir, exp_exp_name), pfn_ids, "HPOBench %s: %s\n Summary over %d datasets and %d trials" % (hpobench_task, exp_exp_name, n_functions, n_trials))


def run_hpob(
    n_trials: int,
    bo_iters: int,
    seed: int,
    baselines: list,
    pfns: list,
    hpob_n_tasks: int,
    n_functions: int,
    exp_dir: str,
    cache_dir: str,
    rerun: bool,
    standardize=True,
    hpo_n_features=3,
    **kwargs,
):
    torch.manual_seed(seed)

    try:
        test_data = json.load(open("/home/yl9959/mtpfn/datasets/hpob-data/meta-validation-dataset.json", "r"))
    except FileNotFoundError as e:
        test_data = json.load(open("/home/lily_l/private_multitask_pfn/datasets/hpob-data/meta-validation-dataset.json", "r"))
    top_level_cache_dir = os.path.join(cache_dir, "hpob")

    ids = []
    for domain in test_data:
        valid_runs = []
        for hpo_run in test_data[domain]:
            x_shape = torch.tensor(test_data[domain][hpo_run]["X"]).shape
            if x_shape[0] > 512 and x_shape[-1] <= hpo_n_features:
                valid_runs.append(hpo_run)
        if len(valid_runs) >= hpob_n_tasks:
            for i in range(n_functions):
                # randomly select runs
                selected_indices = torch.randperm(len(valid_runs))[:hpob_n_tasks]
                selected_runs = [valid_runs[i] for i in selected_indices]
                ids.append((domain, (selected_runs[0], tuple(selected_runs[1:]))))

    num_samples = [
        (2, 20),
        (5, 20),
        (20, 20),
        (2, 100),
        (20, 100),
    ]

    for n_target, n_source in num_samples:
        param_name = "features_%d__n_target_%d__n_source_%d" % (hpo_n_features, n_target, n_source)
        
        for i, (domain, (target_id, source_ids)) in enumerate(ids):        
            source_ids_str = "_".join([str(id) for id in source_ids])
            
            function_seed = seed + i * FUNCTION_SEED_OFFSET
            
            data_xs = []
            data_ys = []
            for id in [target_id] + list(source_ids):
                xs = torch.tensor(test_data[domain][id]["X"])
                ys = torch.tensor(test_data[domain][id]["y"])
                max_indices = 2000
                if xs.shape[0] > max_indices:
                    indices = torch.randperm(xs.shape[0])[:max_indices]
                    xs = xs[indices]
                    ys = ys[indices]

                data_xs.append(xs)
                data_ys.append(ys)

            target_xs, sources_xs = data_xs[0], data_xs[1:]
            target_ys, sources_ys = data_ys[0], data_ys[1:]
            data = (target_xs, target_ys, sources_xs, sources_ys)
        
            exp_domain_dir = os.path.join(exp_dir, param_name, "domain_%s" % domain, "function_%d" % i)
            os.makedirs(exp_domain_dir, exist_ok=True)
            
            cache_experiment_name = "features_%d__domain_%s__target_%s__source_%s" % (hpo_n_features, domain, target_id, source_ids_str)
            cache_domain_dir = os.path.join(top_level_cache_dir, param_name, cache_experiment_name)
            os.makedirs(cache_domain_dir, exist_ok=True)
                    
            with botorch.manual_seed(seed):
                eval_metrics(
                    n_trials=n_trials,
                    bo_iters=bo_iters,
                    seed=function_seed,
                    baselines=baselines,
                    pfns=pfns,
                    data=data,
                    n_target=n_target,
                    n_source=n_source,
                    exp_dir=exp_domain_dir,
                    cache_dir=cache_domain_dir,
                    rerun=rerun,
                    standardize=standardize,
                )
                
            plot_function(exp_domain_dir, n_trials, "HPO-B: N Target %d, N Source %d\nDomain %s (%d Trials)" % (n_target, n_source, domain, n_trials))
            plot_bo_function(exp_domain_dir, "HPO-B: N Target %d, N Source %d\nDomain %s (%d Trials)" % (n_target, n_source, domain, n_trials))
            
        pfn_ids = [pfn_id for pfn_id, _ in pfns]
        plot_all(os.path.join(exp_dir, param_name), n_functions, "HPO-B: Domain %s\n Summary over %d datasets and %d trials" % (domain, n_functions, n_trials))
        plot_bo_all(os.path.join(exp_dir, param_name), pfn_ids, "HPO-B: Domain %s\n Summary over %d datasets and %d trials" % (domain, n_functions, n_trials))


def run_fcnet(
    n_trials: int,
    bo_iters: int,
    seed: int,
    baselines: list,
    pfns: list,
    n_functions: int,
    fcnet_n_features: int,
    exp_dir: str,
    cache_dir: str,
    rerun: bool,
    standardize=True,
    **kwargs,
):
    num_features = fcnet_n_features
    torch.manual_seed(seed)
    top_level_cache_dir = os.path.join(cache_dir, "fcnet")


    problems = [
        "fcnet_naval_propulsion_data",
        "fcnet_parkinsons_telemonitoring_data",
        "fcnet_protein_structure_data",
        "fcnet_slice_localization_data",
    ]
    try:
        datasets = {
            problem: pickle.load(open(f"/home/yl9959/mtpfn/datasets/fcnet_tabular_benchmarks/{problem}.pkl", "rb"))
            for problem in problems
        }
    except FileNotFoundError as e:
        datasets = {
            problem: pickle.load(open(f"/home/lily_l/private_multitask_pfn/datasets/fcnet_tabular_benchmarks/{problem}.pkl", "rb"))
            for problem in problems
        }
    except Exception as e:
        raise e

    num_samples = [
        (2, 20),
        (5, 20),
        (20, 20),
        (2, 100),
        (20, 100),
    ]
    params = itertools.product(problems, num_samples)

    for n_target, n_source in num_samples:
        param_name = "features_%d__n_target_%d__n_source_%d" % (num_features, n_target, n_source)
        for problem in problems:

            target_result = datasets[problem]
            source_results = [datasets[key] for key in datasets if key != problem]

            target_xs, target_ys, sources_xs, sources_ys = get_torch_format_fcnet(
                target_result, source_results, num_features
            )
            
            exp_problem_dir = os.path.join(exp_dir, param_name, problem)
            os.makedirs(exp_problem_dir, exist_ok=True)
            
            cache_problem_dir = os.path.join(top_level_cache_dir, param_name, problem)
            os.makedirs(cache_problem_dir, exist_ok=True)
            
            with botorch.manual_seed(seed):
                eval_metrics(
                    n_trials=n_trials,
                    bo_iters=bo_iters,
                    seed=seed,
                    baselines=baselines,
                    pfns=pfns,
                    data=(target_xs, target_ys, sources_xs, sources_ys),
                    n_target=n_target,
                    n_source=n_source,
                    exp_dir=exp_problem_dir,
                    cache_dir=cache_problem_dir,
                    rerun=rerun,
                    standardize=standardize,
                )
                
            plot_function(exp_problem_dir, n_trials, "FCNet %s: N Target %d, N Source %d (%d Trials)" % (problem, n_target, n_source, n_trials))
            plot_bo_function(exp_problem_dir, "FCNet %s: N Target %d, N Source %d (%d Trials)" % (problem, n_target, n_source, n_trials))
        
        pfn_ids = [pfn_id for pfn_id, _ in pfns]
        plot_all(os.path.join(exp_dir, param_name), n_functions, "FCNet: N Target %d, N Source %d\n Summary over all datasets and %d trials" % (n_target, n_source, n_trials))
        plot_bo_all(os.path.join(exp_dir, param_name), pfn_ids, "FCNet: N Target %d, N Source %d\n Summary over all datasets and %d trials" % (n_target, n_source, n_trials))


def run_test(
    n_trials: int,
    bo_iters: int,
    seed: int,
    baselines: list,
    pfns: list,
    suite: str,
    n_functions: int,
    exp_dir: str,
    cache_dir: str,
    rerun: bool,
    standardize=True,
    **kwargs,
):
    torch.manual_seed(seed)

    eval_suite = get_eval_suite(suite)
    
    top_level_cache_dir = os.path.join(cache_dir, "synthetic")

    results = {}
    for param in eval_suite:
        if param in results:
            continue
        
        results[param] = {}
        (
            num_features,
            num_tasks,
            lengthscale,
            task_corr,
            num_uncorr_tasks,
            (n_target, n_source),
        ) = param
        
        param_key = "features_%d__tasks_%d__lengthscale_%s__task_corr_%s__uncorr_tasks_%d__n_target_%d__n_source_%d" % (
            num_features,
            num_tasks,
            str(lengthscale),
            str(task_corr),
            num_uncorr_tasks,
            n_target,
            n_source,
        )
        exp_title = "Synthetic Data: Features: %d, Tasks: %d\nLengthscale: %s, Task Corr: %s, Uncorr Tasks: %d\nN Target: %d, N Source: %d" % (
            num_features,
            num_tasks,
            str(lengthscale),
            str(task_corr),
            num_uncorr_tasks,
            n_target,
            n_source,
        )
        # replace special characters
        param_dirname = param_key.replace("[", "").replace("]", "").replace(".", "_")
        seed_dirname = "seed_%d" % seed
        
        for function in range(n_functions):
            function_seed = seed + function * FUNCTION_SEED_OFFSET
            exp_function_dir = os.path.join(exp_dir, param_dirname + "__" + seed_dirname, "function_%d" % function)
            os.makedirs(exp_function_dir, exist_ok=True)
            
            cache_function_dir = os.path.join(top_level_cache_dir, param_dirname, seed_dirname, "function_%d" % function)
            os.makedirs(cache_function_dir, exist_ok=True)

            with botorch.manual_seed(function_seed):
                data = get_mtgp_for_eval(
                    num_features,
                    num_tasks,
                    lengthscale,
                    task_corr,
                    num_uncorr_tasks,
                )
                # break
                
            eval_metrics(
                n_trials=n_trials,
                bo_iters=bo_iters,
                seed=function_seed,
                baselines=baselines,
                pfns=pfns,
                data=data,
                n_target=n_target,
                n_source=n_source,
                exp_dir=exp_function_dir,
                cache_dir=cache_function_dir,
                rerun=rerun,
                standardize=standardize,
            )
            
            plot_function(exp_function_dir, n_trials, exp_title + "\nDataset %d (%d Trials)" % (function + 1, n_trials))
            plot_bo_function(exp_function_dir, exp_title + "\nDataset %d (%d Trials)" % (function + 1, n_trials))
            # break
            
        pfn_ids = [pfn_id for pfn_id, _ in pfns]
        plot_all(os.path.join(exp_dir, param_dirname + "__" + seed_dirname), n_functions, exp_title + "\n Summary over %d datasets and %d trials" % (n_functions, n_trials))
        plot_bo_all(os.path.join(exp_dir, param_dirname + "__" + seed_dirname), pfn_ids, exp_title + "\n Summary over %d datasets and %d trials" % (n_functions, n_trials))
        # break


# main function
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--benchmark", type=str, default="synthetic", help="{synthetic, hpobench, hpob, fcnet}")
    argparser.add_argument("--bo_iters", type=int, default=20)
    argparser.add_argument("--n_trials", type=int, default=5)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--n_functions", type=int, default=3)
    
    # baselines
    argparser.add_argument("--baselines", nargs="*", type=str, help="{mtgp, gp, scaml, lmc, mtgp_nuts}", 
                           default=["mtgp", "gp", "scaml", "random"])
    
    # pfn wandb ids
    argparser.add_argument("--pfn_ids", nargs="*", type=str, help="wandb ids for pfn models")
    argparser.add_argument("--ckpt_dirs", nargs="*", type=str, help="directories")
    
    # synthetic
    argparser.add_argument("--suite", type=str, default="corr", help="{num_samples, corr, uncorr, num_tasks}")
    
    # hpobench
    argparser.add_argument("--hpobench_task", type=str, default="rf", help="{rf, xgb, svm, mlp, nn}")
    argparser.add_argument("--hpobench_n_tasks", type=int, default=4)
    argparser.add_argument("--hpobench_n_features", type=int, default=3)
    
    # hpob
    argparser.add_argument("--hpob_n_tasks", type=int, default=4)
    argparser.add_argument("--hpob_n_features", type=int, default=3)
    
    # fcnet
    argparser.add_argument("--fcnet_n_features", type=int, default=3)
    
    argparser.add_argument("--rerun", action="store_true", default=False)
    
    if os.path.exists("/home/yl9959/mtpfn/"):
        home_dir = "/home/yl9959/mtpfn/"
    else:
        home_dir = "/home/lily_l/private_multitask_pfn/"
    argparser.add_argument("--home_dir", type=str, default=home_dir)
    # argparser.add_argument("--exp_dir", type=str, default="/home/yl9959/mtpfn/eval_plot/")
    # argparser.add_argument("--cache_dir", type=str, default="/home/yl9959/mtpfn/eval_plot_cache/")

    args = argparser.parse_args()
    
    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    dirname = current_time + "__" + args.benchmark + "__trials_%d__seed_%d" % (args.n_trials, args.seed)
    
    if args.benchmark == "synthetic":
        dirname += "__" + args.suite
    elif args.benchmark == "hpobench":
        dirname += "__" + args.hpobench_task
        
    args.exp_dir = os.path.join(args.home_dir, "eval_plot", dirname)
    os.makedirs(args.exp_dir, exist_ok=True)
    args.cache_dir = os.path.join(args.home_dir, "eval_plot_cache")
    
    # save args json
    with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
        
        
    pfns = []
    if args.pfn_ids is not None:
        for pfn_id in args.pfn_ids:
            try:
                pfn_dir = os.path.join(args.home_dir, "wandb_links", pfn_id)
                model = load_model(pfn_dir)
                pfns.append((pfn_id, model))
            except FileNotFoundError as e:
                pfn_dir = os.path.join(args.home_dir, "final_models", pfn_id)
                model = load_model(pfn_dir)
                pfns.append((pfn_id, model))
            except Exception as e:
                traceback.print_exc()
                print("failed to load", pfn_id)
                continue
    if args.ckpt_dirs is not None:
        for pfn_dir in args.ckpt_dirs:
            try:
                model = load_model(os.path.join(args.home_dir, pfn_dir))
                pfns.append((pfn_dir, model))
            except FileNotFoundError as e:
                print("failed to load", pfn_dir)
                continue
    
    if args.benchmark == "synthetic":
        run_test(**vars(args), pfns=pfns)
    elif args.benchmark == "hpobench":
        run_hpobench(**vars(args), pfns=pfns)
    elif args.benchmark == "hpob":
        run_hpob(**vars(args), pfns=pfns)
    elif args.benchmark == "fcnet":
        run_fcnet(**vars(args), pfns=pfns)
    else:
        raise ValueError("Invalid benchmark")
    
# python eval.py --benchmark synthetic --bo_iters 10 --n_trials 10 --n_functions 3 --suite uncorr --pfn_ids treasured-lion-279 royal-firebrand-281 different-dust-282 exalted-wave-278 peach-plasma-280