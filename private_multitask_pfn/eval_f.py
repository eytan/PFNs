import botorch
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.multitask import MultiTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors import Posterior
from fblearner.flow import api as flow
from fblearner.flow.api import ResourceRequirements
from fblearner.flow.external_api import WorkflowRun
from fblearner.flow.projects.ae.benchmarks.pfn.thirdparty.PFNs.pfns.bar_distribution import (
    BarDistribution,
)
from fblearner.flow.projects.ae.benchmarks.pfn.utils import (
    load_model,
    to_gp_format,
    to_mtgp_format,
    to_pfn_format,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch.distributions import Uniform


def branin_source(x):
    x1, x2 = x[..., 0], x[..., 1]
    a, b, c, r, s, t = (
        Uniform(
            torch.tensor([0.5, 0.1, 1, 5, 8, 0.03]),
            torch.tensor([1.5, 0.15, 2, 7, 12, 0.05]),
        )
        .sample()
        .tolist()
    )
    first = a * (x2 - b * x1**2 + c * x1 - r)
    second = s * (1 - t) * torch.cos(x1) + s
    return first + second


def hartmann_source(x):
    alphas = (
        Uniform(
            torch.tensor([1.0, 1.18, 2.8, 3.2]), torch.tensor([1.02, 1.20, 3.0, 3.4])
        )
        .sample()
        .to(device=x.device)
    )
    A = torch.tensor([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]).to(
        device=x.device
    )
    P = (
        torch.tensor(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        ).to(device=x.device)
        * 1e-4
    )
    inner = -(A * (x.unsqueeze(-2) - P) ** 2)
    # return negative for maximization
    return (alphas * torch.exp(inner.sum(-1))).sum(-1).unsqueeze(-1)


def pfn_gaussian_get_best(pfn, train_x, train_y, bounds):
    target_indices = train_x[..., 0] > 0
    best_f = train_y[target_indices].max()

    pfn_model = PFNGaussian(pfn, train_x, train_y)
    ei = ExpectedImprovement(pfn_model, best_f=best_f)
    candidates, _ = optimize_acqf(
        ei, bounds=bounds, q=1, num_restarts=10, raw_samples=512
    )
    return candidates


def gp_get_best(train_x, train_y, bounds):
    train_x, train_y = from_pfn_to_gp_format(train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5

    best_f = train_y.max()

    gp = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=None,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    ei = ExpectedImprovement(gp, best_f=best_f)
    candidates, _ = optimize_acqf(
        ei, bounds=bounds, q=1, num_restarts=10, raw_samples=512
    )
    return candidates


def pfn_bar_get_best(pfn, train_x, train_y, bounds):
    target_indices = train_x[..., 0] > 0
    logits = pfn(train_x, train_y, possible_x)
    eis = pfn.criterion.ei(logits, train_y[target_indices].max())
    best_index = eis.argmax()
    return possible_x[best_index], possible_y[best_index]


def pfn_get_best(pfn, train_x, train_y, bounds):
    if isinstance(pfn.criterion, BarDistribution):
        return pfn_bar_get_best(pfn, train_x, train_y, bounds)
    else:
        return pfn_gaussian_get_best(pfn, train_x, train_y, bounds)


def mtgp_get_best(task_id, train_x, train_y, bounds):
    train_x, train_y = to_mtgp_format(task_id, train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5

    best_f = train_y.max()

    mtgp = MultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        train_Yvar=train_yvar,
        outcome_transform=None,
        output_tasks=[0],
    )
    mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
    fit_gpytorch_mll(mll)

    ei = ExpectedImprovement(mtgp, best_f=best_f)
    candidates, _ = optimize_acqf(
        ei, bounds=bounds, q=1, num_restarts=10, raw_samples=512
    )
    return candidates


# @flow.flow_async(resource_requirements=ResourceRequirements(cpu=1))
# @flow.typed()
def bo_loop_continuous(
    model_type, f, init_id, init_x, init_y, iters, bounds, model=None
):
    print("STARTING", model_type)
    train_x = init_x.clone()
    train_y = init_y.clone()

    for i in range(iters):
        if model_type == "mtgp":
            best_x = mtgp_get_best_continuous(train_id, train_x, train_y, bounds)
        elif model_type == "gp":
            best_x = gp_get_best_continuous(train_id, train_x, train_y, bounds)
        elif model_type == "pfn":
            if model is None:
                raise ValueError("model must be provided for pfn")
            best_x = pfn_get_best_continuous(model, train_id, train_x, train_y, bounds)
        else:
            raise ValueError("model_type must be one of ['pfn', 'mtgp', 'gp']")

        best_y = f(best_x)

        n_tasks = train_x.shape[-1] - best_x.shape[-1]
        best_x_pad = torch.zeros(*best_x.shape[:-1], n_tasks, device=best_x.device)
        best_x_pad[..., 0] = 1
        best_x_pad = torch.cat((best_x_pad, best_x), -1)

        # # pad with zeros and 1 hot target encoding
        # n_features = train_x.shape[-1]
        # best_x_pad = torch.zeros(1, n_features, device=train_x.device)
        # best_x_pad[:, 0] = 1
        # best_x_pad[:, -best_x.shape[-1] :] = best_x

        train_x = torch.cat((train_x, best_x_pad.unsqueeze(0)), 0)
        train_y = torch.cat((train_y, best_y.unsqueeze(0)), 0)

    return train_x, train_y


@flow.flow_async(resource_requirements=ResourceRequirements(cpu=1, gpu=1, memory="16g"))
@flow.typed()
def run_bo_loop_continuous(
    f, train_id, train_x, train_y, num_features, bo_iters, run_ids
):
    train_id.to(device="cuda")
    train_x.to(device="cuda")
    train_y.to(device="cuda")

    info = {}
    pfns = []
    for run in run_ids:
        workflow_run = WorkflowRun(run)
        results = workflow_run.get_results()["output"]
        pfn = load_model(results, num_tasks=4)
        pfns.append(pfn)
        info[run] = results["config"]

    results = {}

    bounds = torch.stack(
        [
            torch.zeros(num_features, device=train_x.device),
            torch.ones(num_features, device=train_x.device),
        ]
    )

    results["mtgp"] = bo_loop_continuous(
        "mtgp", f, train_id, train_x, train_y, bo_iters, bounds
    )
    results["gp"] = bo_loop_continuous(
        "gp", f, train_id, train_x, train_y, bo_iters, bounds
    )
    for run_id, pfn in zip(run_ids, pfns):
        results[run_id] = bo_loop_continuous(
            "pfn", f, train_id, train_x, train_y, bo_iters, bounds, pfn
        )

    return {
        "results": results,
        "info": info,
    }


@flow.registered(owners=["oncall+ae"])
@flow.typed()
def run_hartmann(runs: list, n_trials, bo_iters, num_tasks=4, seed=0, device="cpu"):
    torch.manual_seed(seed)

    def hartmann(x):
        return botorch.test_functions.Hartmann(dim=3, negate=True)(
            x[..., -3:]
        ).unsqueeze(-1)

    num_features = 3
    max_num_tasks = 4

    params = [
        ("hartmann", 1, 20),
        ("hartmann", 5, 20),
        ("hartmann", 5, 50),
        ("hartmann", 20, 50),
    ]

    results = {}
    for param in params:
        print("RUNNING", param)
        results[param] = {}
        name, num_target, num_source = param

        for trial in range(n_trials):
            train_id = []
            train_x = []
            train_y = []

            for i in range(num_tasks):
                task_x = torch.rand(num_source, num_features, device=device)

                if i == 0:
                    task_id = torch.zeros(num_target)
                    task_y = hartmann(task_x)
                else:
                    task_id = torch.ones(num_source) * i
                    task_y = hartmann_source(task_x)

                train_id.append(task_id)
                train_x.append(task_x)
                train_y.append(task_y)

            train_id = torch.cat(train_id)
            train_x = torch.cat(train_x)
            train_y = torch.cat(train_y)

            result = run_bo_loop_continuous(
                hartmann,
                train_id,
                train_x,
                train_y,
                num_features,
                bo_iters=bo_iters,
                run_ids=runs,
            )
            results[param][trial] = result
    return results
