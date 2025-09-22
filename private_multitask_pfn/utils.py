import gc

import botorch
import torch
# from ax.fb.utils.storage.manifold import AEManifoldUseCase
# from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll

from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.multitask import MultiTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors import Posterior
from botorch.utils.datasets import SupervisedDataset
# from botorch_fb.experimental.models.scaml import meta_fit_scamlgp, ScaMLGP
from scaml import meta_fit_scamlgp, ScaMLGP
from PFNs.pfns.bar_distribution import BarDistribution

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from train import train as load_model_from_train
import json
from lmc import LMCGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from mtgp_nuts import RBFPyroModel, MultitaskRBFPyroModel
from gpytorch.kernels import RBFKernel
from gpytorch.priors import GammaPrior


class GaussianPosterior(Posterior):
    def __init__(self, mean, variance):
        super().__init__()
        self.mean = mean
        self.variance = variance

    def rsample(self, sample_shape):
        return self.mean + torch.sqrt(self.variance) * torch.randn(
            sample_shape, device=self.mean.device
        )
        
    @property
    def lower(self):
        return self.mean - 2 * torch.sqrt(self.variance)
    
    @property
    def upper(self):
        return self.mean + 2 * torch.sqrt(self.variance)

    @property
    def dtype(self):
        return self.mean.dtype

    @property
    def device(self):
        return self.mean.device
    
    
class BarPosterior(Posterior):
    def __init__(self, logits, criterion):
        super().__init__()
        self.logits = logits
        self.criterion = criterion

    def rsample(self, sample_shape):
        probs = torch.rand(sample_shape)
        return self.criterion.icdf(self.logits, probs)
        
    @property
    def lower(self):
        return self.criterion.quantile(self.logits, 0.025)[..., 0]
    
    @property
    def upper(self):
        return self.criterion.quantile(self.logits, 0.975)[..., 0]
    
    @property
    def mean(self):
        return self.criterion.mean(self.logits)
    
    @property
    def variance(self):
        variance = torch.clamp_min(self.criterion.variance(self.logits), 1e-8)
        return variance

    @property
    def dtype(self):
        return self.logits.dtype

    @property
    def device(self):
        return self.logits.device


class PFNGaussian(Model):
    def __init__(self, pfn, train_x, train_task_id, train_y):
        super().__init__()
        self.pfn = pfn
        self.train_x = train_x.unsqueeze(1)
        self.train_task_id = train_task_id.unsqueeze(-1).unsqueeze(1).long()
        self.train_y = train_y.unsqueeze(1)

    def posterior(
        self, X, output_indices=None, observation_noise=False, posterior_transform=None
    ):
        original_shape = X.shape
        X_reshape = X.reshape(X.shape[0], -1, X.shape[-1])
        pfn_outputs = self.pfn(self.train_x, self.train_task_id, self.train_y, X_reshape, None)
        mean = pfn_outputs[..., 0]
        variance = pfn_outputs[..., 1].exp()
        
        mean = mean.reshape(original_shape[:-1])
        variance = variance.reshape(original_shape[:-1])
        return GaussianPosterior(mean, variance)

    @property
    def num_outputs(self):
        return 1


def load_model(ckpt_dir, best=True):
    args_json = f"{ckpt_dir}/args.json"
    with open(args_json, "r") as f:
        args = json.load(f)

    model = load_model_from_train(**args, return_model=True)[0]
    default_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if best:
        try:
            model.load_state_dict(torch.load(f"{ckpt_dir}/best_model.pth", weights_only=True, map_location=default_device()))
        except FileNotFoundError:
            model.load_state_dict(torch.load(f"{ckpt_dir}/final_model.pth", weights_only=True, map_location=default_device()))
        except RuntimeError:
            try:
                new_state_dict = torch.load(f"{ckpt_dir}/best_model.pth", map_location=default_device())
                new_state_dict_keys = new_state_dict.keys()
                new_state_dict = {k.replace("module.", ""): v for k, v in new_state_dict.items()}
                model.load_state_dict(new_state_dict, strict=True)
            except Exception as e:
                raise e
    else:
        model.load_state_dict(torch.load(f"{ckpt_dir}/final_model.pth", weights_only=True, map_location=default_device()))
    
    print("Loaded model")
    return model


def load_model_from_epoch(ckpt_dir, epoch):
    args_json = f"{ckpt_dir}/args.json"
    with open(args_json, "r") as f:
        args = json.load(f)

    model = load_model_from_train(**args, return_model=True)[0]
    default_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"{ckpt_dir}/model_{epoch}.pth", weights_only=True, map_location=default_device()))

    print("Loaded model")
    return model


def load_checkpoint(ckpt_dir):
    
    args_json = f"{ckpt_dir}/args.json"
    with open(args_json, "r") as f:
        args = json.load(f)

    model, scheduler, optimizer = load_model_from_train(**args, return_model=True)
    
    ckpt = torch.load(f"{ckpt_dir}/checkpoint.pth")
    model.load_state_dict(ckpt['model_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    print("Loaded checkpoint")
    return model, scheduler, optimizer, ckpt['epoch']


def to_pfn_format(num_tasks, *args):
    """
    Expects task1, x1, y1, task2, x2, y2, ...
    where x1, x2, ... are [seq, features]

    Returns x, y, where x is [batch, seq, features + tasks] and y is [batch, seq]
    """
    assert len(args) % 3 == 0 and len(args) > 0

    results = []
    for i in range(len(args) // 3):
        task_id, x, y = args[3 * i], args[3 * i + 1], args[3 * i + 2]

        one_hot_task_id = torch.nn.functional.one_hot(
            task_id.long(), num_classes=num_tasks
        )
        id_x = torch.cat((one_hot_task_id, x), -1)

        # add batch dimension
        results.append(id_x.unsqueeze(1))
        results.append(y.unsqueeze(1))

    return results


def to_mtgp_format(*args):
    """
    Expects task1, x1, y1, task2, x2, y2, ...
    where x1, x2, ... are [seq, features]

    Returns x, y, where x is [batch, seq, features + 1] and y is [batch, seq]
    """
    assert len(args) % 3 == 0 and len(args) > 0

    results = []
    for i in range(len(args) // 3):
        task_id, x, y = args[3 * i], args[3 * i + 1], args[3 * i + 2]

        # x is [seq, features]
        # if len(x.shape) > 2:
        #     x = x.transpose(0, 1)
        #     y = y.transpose(0, 1)

        task_id = torch.ones_like(x[..., 0]) * task_id.to(x.device)
        x_task_id = torch.cat((x, task_id.unsqueeze(-1)), -1)
        results.append(x_task_id)
        results.append(y)

    return results


def to_gp_format(*args):
    """
    Expects task1, x1, y1, task2, x2, y2, ...
    where x1, x2, ... are [seq, features]

    Returns x, y, where x is [batch, seq, features] and y is [batch, seq]
    Only over target task (task id = 0)
    """
    assert len(args) % 3 == 0 and len(args) > 0

    results = []
    for i in range(len(args) // 3):
        task_id, x, y = args[3 * i], args[3 * i + 1], args[3 * i + 2]

        # x is [seq, features]
        # if len(x.shape) > 2:
        #     x = x.transpose(0, 1)
        #     y = y.transpose(0, 1)

        target_indices = task_id == 0

        target_x = x[target_indices]
        target_y = y[target_indices]  # .unsqueeze(1)
        results.append(target_x)
        results.append(target_y)
        
    # print([r.shape for r in results])

    return results


def pfn_bar_get_best(
    pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    original_possible_x, original_possible_y = possible_x.clone(), possible_y.clone()
    train_x, train_y, possible_x, possible_y = to_pfn_format(
        pfn.num_tasks,
        task_id,
        train_x,
        train_y,
        possible_task_id,
        possible_x,
        possible_y,
    )
    target_indices = task_id == 0
    logits = pfn(train_x, train_y, possible_x)
    eis = pfn.criterion.ei(logits, train_y[target_indices].max())
    best_index = eis.argmax()
    return original_possible_x[best_index], original_possible_y[best_index]


def pfn_gaussian_fit(pfn, task_id, train_x, train_y):
    return PFNGaussian(pfn, train_x, task_id, train_y)


def pfn_bar_get_best(
    pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    # add batch dimension
    train_x = train_x.unsqueeze(1)
    train_task_id = task_id.unsqueeze(-1).unsqueeze(1).long()
    train_y = train_y.unsqueeze(1)
    possible_x = possible_x.unsqueeze(1)
    
    output_logits = pfn(train_x, train_task_id, train_y, possible_x, None)#.squeeze(1)
    
    target_indices = task_id == 0
    best_f = train_y[target_indices].max()
    
    ei = pfn.criterion.ei(output_logits, best_f)
    best_index = ei.argmax()
    return possible_x[best_index].squeeze(1).squeeze(0), possible_y[best_index]
    


def pfn_gaussian_get_best(
    pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    pfn_model = pfn_gaussian_fit(pfn, task_id, train_x, train_y)

    target_indices = task_id == 0
    best_f = train_y[target_indices].max()

    ei = LogExpectedImprovement(pfn_model, best_f=best_f)
    eis = ei(possible_x.unsqueeze(1))
    
    best_index = eis.argmax()
    return possible_x[best_index], possible_y[best_index]


def pfn_get_best(
    pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    if isinstance(pfn.criterion, BarDistribution):
        return pfn_bar_get_best(
            pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
        )
    else:
        return pfn_gaussian_get_best(
            pfn, task_id, train_x, train_y, possible_task_id, possible_x, possible_y
        )
        
        
def pfn_bar_predict(pfn, task_id, train_x, train_y, test_x):
    train_x = train_x.unsqueeze(1)
    train_task_id = task_id.unsqueeze(-1).unsqueeze(1).long()
    train_y = train_y.unsqueeze(1)
    test_x = test_x.unsqueeze(1)
    
    output_logits = pfn(train_x, train_task_id, train_y, test_x, None)
    # BarPosterior(output_logits, pfn.criterion)
    # mean, variance = pfn.criterion.mean(output_logits), pfn.criterion.variance(output_logits)
    return BarPosterior(output_logits, pfn.criterion)
        
        
def pfn_predict(pfn, task_id, train_x, train_y, possible_x):
    with torch.no_grad():
        if isinstance(pfn.criterion, BarDistribution):
            return pfn_bar_predict(pfn, task_id, train_x, train_y, possible_x)
        else:
            pfn_model = pfn_gaussian_fit(pfn, task_id, train_x, train_y)
            return pfn_model.posterior(possible_x.unsqueeze(1))


def mtgp_fit(task_id, train_x, train_y):
    train_x, train_y = to_mtgp_format(task_id, train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5
    mtgp = MultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        train_Yvar=train_yvar,
        outcome_transform=None,
        output_tasks=[0],
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
    try:
        fit_gpytorch_mll(mll)
    except botorch.exceptions.ModelFittingError:
        fit_gpytorch_mll_torch(mll)

    return mtgp


def mtgp_predict(task_id, train_x, train_y, possible_x):
    mtgp = mtgp_fit(task_id, train_x, train_y)
    possible_x, _ = to_mtgp_format(torch.zeros_like(possible_x[..., 0]), possible_x, torch.zeros_like(possible_x))
    return mtgp.posterior(possible_x)


def mtgp_get_best(task_id, train_x, train_y, possible_task_id, possible_x, possible_y):
    gc.collect()

    original_possible_x, original_possible_y = possible_x.clone(), possible_y.clone()
    train_x, train_y, possible_x, possible_y = to_mtgp_format(
        task_id, train_x, train_y, possible_task_id, possible_x, possible_y
    )
    
    mtgp = mtgp_fit(task_id, train_x, train_y)
    ei = LogExpectedImprovement(mtgp, best_f=train_y[task_id == 0].max())

    batch_size = max(10, 1000 - len(train_x))
    possible_xs = possible_x.unsqueeze(1).split(
        batch_size
    )  # add batch dimension for EI, split for memory
    best_value = -1.0
    best_index = None
    for i, x_batch in enumerate(possible_xs):
        max_value, max_index = ei(x_batch).max(0)
        if max_value > best_value or best_index is None:
            best_index = i * batch_size + max_index
            best_value = max_value

    gc.collect()
    assert best_index is not None

    return original_possible_x[best_index], original_possible_y[best_index]


def gp_fit(task_id, train_x, train_y):
    train_x, train_y = to_gp_format(task_id, train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5
    covar_module = RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0))
    gp = SingleTaskGP(train_x, train_y, train_yvar, outcome_transform=None).to(train_x)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except botorch.exceptions.ModelFittingError:
        fit_gpytorch_mll_torch(mll)

    return gp


def gp_predict(task_id, train_x, train_y, possible_x):
    gp = gp_fit(task_id, train_x, train_y)
    possible_x, _ = to_gp_format(torch.zeros_like(possible_x[..., 0]), possible_x, torch.zeros_like(possible_x))
    return gp.posterior(possible_x)


def gp_get_best(
    train_task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    gp = gp_fit(train_task_id, train_x, train_y)
    possible_x, possible_y = to_gp_format(
        possible_task_id, possible_x, possible_y
    )

    ei = LogExpectedImprovement(gp, best_f=train_y.max())
    best_index = ei(possible_x.unsqueeze(1)).argmax()  # add batch dimension
    return possible_x[best_index], possible_y[best_index]



def scaml_fit(task_id, train_x, train_y):
    meta_data = {}
    for task in torch.unique(task_id):
        if task == 0:
            continue
        source_x = train_x[task_id == task]
        source_y = train_y[task_id == task]
        source_yvar = torch.ones_like(source_y) * 1e-5
        dataset = SupervisedDataset(
            source_x,
            source_y,
            feature_names=["x"] * source_x.shape[-1],
            outcome_names=["y"],
            Yvar=source_yvar,
        )
        meta_data[task] = dataset

    source_gps = meta_fit_scamlgp(meta_data)

    target_x = train_x[task_id == 0]
    target_y = train_y[task_id == 0]
    target_yvar = torch.ones_like(target_y) * 1e-5

    gp = ScaMLGP(
        target_x,
        target_y,
        source_gps=source_gps,
        likelihood=FixedNoiseGaussianLikelihood(target_yvar),
    ).to(target_x)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return gp


def scaml_predict(task_id, train_x, train_y, possible_x):
    gp = scaml_fit(task_id, train_x, train_y)
    return gp.posterior(possible_x.to(train_x))


def scaml_get_best(
    train_task_id, train_x, train_y, possible_task_id, possible_x, possible_y
):
    gp = scaml_fit(train_task_id, train_x, train_y)
    target_y = train_y[train_task_id == 0]

    ei = LogExpectedImprovement(gp, best_f=target_y.max())
    best_index = ei(possible_x.unsqueeze(1).to(train_x)).argmax()  # add batch dimension
    return possible_x[best_index], possible_y[best_index]


def standardize_by_task(train_y, task_id, test_y=None):
    standardized_y = train_y.clone()
    if test_y is not None:
        std_test_y = test_y.clone()
    for task in torch.unique(task_id):
        task_mask = task_id == task
        task_ys = train_y[task_mask]

        std = task_ys.std(0)
        std = torch.where(std.isnan(), torch.tensor(1.0), std)
        std = torch.where(std < 1e-5, torch.tensor(1.0), std)
        standardized_y[task_mask] = (task_ys - task_ys.mean(0)) / std
        if test_y is not None and task == 0:
            std_test_y = (test_y - task_ys.mean(0)) / std

    if test_y is not None:
        return standardized_y, std_test_y
    return standardized_y


def create_train_test(target_xs, target_ys, sources_xs, sources_ys, n_target, n_source):
    train_id = []
    train_x = []
    train_y = []
    for i, (source_xs, source_ys) in enumerate(zip(sources_xs, sources_ys)):
        random_indices = torch.randperm(len(source_xs))[:n_source]
        train_x.append(source_xs[random_indices])
        train_y.append(source_ys[random_indices])
        train_id.append(torch.ones(n_source) * i + 1)

    # add target task
    random_indices = torch.randperm(len(target_xs))
    train_indices = random_indices[:n_target]
    test_indices = random_indices[n_target:]
    train_x.append(target_xs[train_indices])
    train_y.append(target_ys[train_indices])
    train_id.append(torch.zeros(n_target))

    train_x = torch.concat(train_x, 0)
    train_id = torch.concat(train_id, 0)

    train_y = torch.concat(train_y, 0)

    test_id = torch.zeros(len(test_indices))
    test_x = target_xs[test_indices]
    test_y = target_ys[test_indices]

    return (
        train_id,
        train_x,
        train_y,
        test_id,
        test_x,
        test_y,
    )


def lmc_fit(task_id, train_x, train_y):
    train_x, train_y = to_mtgp_format(task_id, train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5
    num_tasks = len(torch.unique(task_id))
    
    best_mll = None
    best_mtgp = None
    
    for i in range(1, num_tasks):
        # Create model with custom kernel
        lmc = LMCGP(
            train_x,
            train_y,
            task_feature=-1,
            num_covars=i,
            train_Yvar=train_yvar,
            outcome_transform=None,
            output_tasks=[0],
        )
        mll = ExactMarginalLogLikelihood(lmc.likelihood, lmc)
        try:
            fit_gpytorch_mll(mll)
        except botorch.exceptions.ModelFittingError:
            fit_gpytorch_mll_torch(mll)
            
        mll_value = mll(lmc(train_x), train_y).sum()
        if best_mll is None or mll_value > best_mll:
            best_mll = mll_value
            best_mtgp = lmc

    return best_mtgp


def lmc_predict(task_id, train_x, train_y, possible_x):
    lmc = lmc_fit(task_id, train_x, train_y)
    possible_x, _ = to_mtgp_format(torch.zeros_like(possible_x[..., 0]), possible_x, torch.zeros_like(possible_x))
    return lmc.posterior(possible_x)


def lmc_get_best(task_id, train_x, train_y, possible_task_id, possible_x, possible_y):
    gc.collect()

    original_possible_x, original_possible_y = possible_x.clone(), possible_y.clone()
    train_x, train_y, possible_x, possible_y = to_mtgp_format(
        task_id, train_x, train_y, possible_task_id, possible_x, possible_y
    )
    
    lmc = lmc_fit(task_id, train_x, train_y)
    ei = LogExpectedImprovement(lmc, best_f=train_y[task_id == 0].max())

    batch_size = max(10, 1000 - len(train_x))
    possible_xs = possible_x.unsqueeze(1).split(
        batch_size
    )  # add batch dimension for EI, split for memory
    best_value = -1.0
    best_index = None
    for i, x_batch in enumerate(possible_xs):
        max_value, max_index = ei(x_batch).max(0)
        if max_value > best_value or best_index is None:
            best_index = i * batch_size + max_index
            best_value = max_value

    gc.collect()
    assert best_index is not None

    return original_possible_x[best_index], original_possible_y[best_index]


def mtgp_nuts_fit(task_id, train_x, train_y):
    train_x, train_y = to_mtgp_format(task_id, train_x, train_y)
    train_yvar = torch.ones_like(train_y) * 1e-5
    mtgp = SaasFullyBayesianMultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        train_Yvar=train_yvar,
        outcome_transform=botorch.models.transforms.outcome.Standardize(m=1),
        output_tasks=[0],
        pyro_model=MultitaskRBFPyroModel(),
    )
    try:
        fit_fully_bayesian_model_nuts(mtgp)
    except RuntimeError:
        train_yvar = torch.ones_like(train_y) * 1e-4
        mtgp = SaasFullyBayesianMultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            train_Yvar=train_yvar,
            outcome_transform=botorch.models.transforms.outcome.Standardize(m=1),
            output_tasks=[0],
            pyro_model=MultitaskRBFPyroModel(),
        )
        fit_fully_bayesian_model_nuts(mtgp)

    return mtgp


def mtgp_nuts_predict(task_id, train_x, train_y, possible_x):
    mtgp = mtgp_nuts_fit(task_id, train_x, train_y)
    possible_x, _ = to_mtgp_format(torch.zeros_like(possible_x[..., 0]), possible_x, torch.zeros_like(possible_x))
    return mtgp.posterior(possible_x)


def mtgp_nuts_get_best(task_id, train_x, train_y, possible_task_id, possible_x, possible_y):
    gc.collect()

    original_possible_x, original_possible_y = possible_x.clone(), possible_y.clone()
    train_x, train_y, possible_x, possible_y = to_mtgp_format(
        task_id, train_x, train_y, possible_task_id, possible_x, possible_y
    )
    
    mtgp = mtgp_nuts_fit(task_id, train_x, train_y)
    ei = LogExpectedImprovement(mtgp, best_f=train_y[task_id == 0].max())

    batch_size = max(10, 1000 - len(train_x))
    possible_xs = possible_x.unsqueeze(1).split(
        batch_size
    )  # add batch dimension for EI, split for memory
    best_value = -1.0
    best_index = None
    for i, x_batch in enumerate(possible_xs):
        max_value, max_index = ei(x_batch).max(0)
        if max_value > best_value or best_index is None:
            best_index = i * batch_size + max_index
            best_value = max_value

    gc.collect()
    assert best_index is not None

    return original_possible_x[best_index], original_possible_y[best_index]


def tabpfn_fit(task_id, train_x, train_y):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    # onehot task_id
    onehot = torch.nn.functional.one_hot(task_id.long(), num_classes=task_id.max() + 1)
    train_x = torch.cat((onehot, train_x), -1)
    train_y = train_y.unsqueeze(1)
    
    tabpfn = TabPFNRegressor(model_path="/home/lily_l/private_multitask_pfn/TabPFN/tabpfn-v2-regressor.ckpt").fit(train_x, train_y)
    # # https://github.com/PriorLabs/TabPFN/blob/c8959619f0b6e62614c5a1aaa8f41c0a5ac725d3/src/tabpfn/regressor.py#L84
    # output = tabpfn.predict(train_x, output_type="full")
    # quantiles = tabpfn.predict(train_x, output_type="quantiles", quantiles=[0.025, 0.5, 0.975])
    
    # !pip install tabpfn
    return tabpfn
    
def tabpfn_predict(task_id, train_x, train_y, possible_x):
    tabpfn = tabpfn_fit(task_id, train_x, train_y)
    
    test_task_id = torch.zeros(possible_x.size(0)).long()
    onehot = torch.nn.functional.one_hot(test_task_id.long(), num_classes=task_id.max() + 1)
    possible_x = torch.cat((onehot, possible_x), -1)
    # output = tabpfn.predict(possible_x, output_type="full")
    quantiles = tabpfn.predict(possible_x, output_type="quantiles", quantiles=[0.025, 0.5, 0.975])

    return quantiles
    



def plot_multitask(ax, test_info, model_dim):
    train_x, train_task_id, train_y, test_x, test_task_id, test_y = test_info
    
    if model_dim > 1:
        # pad train and test with 0s
        padded_train_x = torch.cat([train_x, torch.zeros(train_x.size(0), model_dim - 1)], dim=1)
        padded_test_x = torch.cat([test_x, torch.zeros(test_x.size(0), model_dim - 1)], dim=1)
    else:
        padded_train_x = train_x
        padded_test_x = test_x
        
    # train_x, train_task_id, train_y, test_x, 
    lower, median, upper = tabpfn_predict(train_task_id, padded_train_x, train_y, padded_test_x)
    
    ax.plot(test_x, test_y, label="true", color="C0")
    ax.plot(test_x, median, label="mean", color="C1")
    ax.fill_between(test_x.flatten(), lower, upper, alpha=0.2, color="C1")
    # ax.plot(test_x, mean, label="mean", color="C1")
    # ax.fill_between(test_x.flatten(), mean.flatten() - 2 * std.flatten(), mean.flatten() + 2 * std.flatten(), alpha=0.2, color="C1")
    
    for i in train_task_id.unique():
        mask = train_task_id == i
        marker = "x" if i.item() == 0 else "o"
        size = 100 if i.item() == 0 else 50
        ax.scatter(train_x[mask], train_y[mask], label=f"train task {i.item()}", color=f"C{i}", marker=marker, s=size)


def get_multitask_test_function(seed=0):
    from gpytorch.kernels import RBFKernel
    from torch.distributions import MultivariateNormal
    
    n_features = 1
    n_tasks = 3
    with botorch.manual_seed(seed):
        # n_samples = torch.randint(10, 40, (1,)).item()
        n_samples = 50
        
        train_xs = torch.rand(n_samples, n_features)
        train_task_id = torch.randint(n_tasks, size=(n_samples,)).unsqueeze(1).long()
        test_xs = torch.linspace(0, 1, 100).view(-1, n_features)
        test_task_id = torch.zeros(test_xs.size(0), 1).long()
        xs = torch.cat([train_xs, test_xs], dim=0)
        task_id = torch.cat([train_task_id, test_task_id], dim=0)
        
        rbf = RBFKernel()
        rbf.lengthscale = torch.tensor(0.2)
        covar_x = rbf(xs)
        task_covar_matrix = torch.ones(n_tasks, n_tasks) * 0.9
        task_covar_matrix += torch.eye(n_tasks) * 0.1
        
        covar_t = task_covar_matrix[task_id, task_id.t()].squeeze()
        covar = covar_x.mul(covar_t).evaluate()
        covar = covar + 1e-4 * torch.eye(covar.size(0))
        ys = MultivariateNormal(torch.zeros(len(covar)), covar).sample()

        train_y = ys[:n_samples]
        test_y = ys[n_samples:]
        
    return train_xs, train_task_id.squeeze(), train_y, test_xs, test_task_id.squeeze(), test_y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seed = 0
    test_info = get_multitask_test_function(seed)
    fig, ax = plt.subplots()
    plot_multitask(ax, test_info, 1)
    plt.savefig("tabpfn.png")
