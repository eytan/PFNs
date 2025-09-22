from utils import load_model, pfn_predict

import numpy as np
import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP

from botorch.models.multitask import MultiTaskGP

from botorch.models.transforms.input import (
    ChainedInputTransform,
    # LatentCategoricalEmbedding,
    # LatentCategoricalSpec,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize, StratifiedStandardize
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Normal
import pickle

import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

def eval_model(model, test_X, test_Y):
    with torch.no_grad():
        posterior = model.posterior(test_X, observation_noise=True)
        # compute sum of LL of each point in test set (using only marginal variances)
        var = posterior.variance
        mean = posterior.mean
        nll = (
            -Normal(loc=mean.squeeze(-1), scale=var.squeeze(-1))
            .log_prob(test_Y.view(-1))
            .sum(dim=-1)
            .mean()  # take average over MCMC samples (if needed)
            .item()
        )
        mse = (mean - test_Y).pow(2).mean().item()
    return nll, mse

def eval_models_on_problem(problem, train_X, train_Y, test_X, test_Y):
    res = {}
    # test STGP on target task
    target_mask = train_X[:, -1] == 0
    model = SingleTaskGP(
        train_X[target_mask],
        train_Y[target_mask],
        input_transform=Normalize(
            d=train_X.shape[-1], indices=list(range(train_X.shape[-1] - 1))
        ),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _ = fit_gpytorch_mll(mll)
    nll, mse = eval_model(model=model, test_X=test_X, test_Y=test_Y)
    res["STGP - target only"] = {"NLL": nll, "MSE": mse}

    # Test MTGP with ICM
    model = MultiTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(
            d=train_X.shape[-1], indices=list(range(train_X.shape[-1] - 1))
        ),
        outcome_transform=StratifiedStandardize(
            stratification_idx=problem.dim - 1,
            task_values=torch.tensor(problem.fidelities, dtype=torch.long),
        ),
        task_feature=problem.dim - 1,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _ = fit_gpytorch_mll(mll)
    nll, mse = eval_model(model=model, test_X=test_X, test_Y=test_Y)
    res["MTGP - ICM - MAP"] = {"NLL": nll, "MSE": mse}
    # Test Fully Bayesian MTGP with Latent Embeddings
    model = SaasFullyBayesianMultiTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(
            d=train_X.shape[-1], indices=list(range(train_X.shape[-1] - 1))
        ),
        task_feature=problem.dim - 1,
        outcome_transform=StratifiedStandardize(
            stratification_idx=problem.dim - 1,
            task_values=torch.tensor(problem.fidelities, dtype=torch.long),
        ),
    )
    _ = fit_fully_bayesian_model_nuts(model, jit_compile=True)
    nll, mse = eval_model(model=model, test_X=test_X, test_Y=test_Y)
    res["MTGP - Latent Embeddings - FB"] = {"NLL": nll, "MSE": mse}

    # test LVGP with MAP estimation
    d = train_X.shape[-1]
    cat_dims = [problem.dim - 1]
    # construct input transform
    input_transform = ChainedInputTransform(
        normalize=Normalize(d=d, indices=list(range(d - 1))),
        # latent_emb=LatentCategoricalEmbedding(
        #     [
        #         LatentCategoricalSpec(
        #             idx=i,
        #             num_categories=len(problem.fidelities),
        #             latent_dim=2,
        #         )
        #         for i in cat_dims
        #     ],
        #     dim=d,
        # ).to(train_X),
    )
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=input_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _ = fit_gpytorch_mll(mll)
    nll, mse = eval_model(model=model, test_X=test_X, test_Y=test_Y)
    res["MTGP - Latent Embeddings - MAP"] = {"NLL": nll, "MSE": mse}
    return res

class WingWeightMultiFidelitySmall(SyntheticTestFunction):
    """Wing Weight Design Problem from [Chen2024]_.

    Design variables (physical units):
      1. s_w   in [150,   200]   (wing area)
      2. w_fw  in [220,   300]   (fuel weight)
      3. A     in [6,     10]    (aspect ratio)
      4. Lambda_deg in [-10, 10]  (sweep angle, degrees)
      5. q     in [16,    45]    (dynamic pressure)
      6. lam   in [0.5,   1.0]   (taper ratio)
      7. t_c   in [0.08,  0.18]   (thickness-to-chord)
      8. N_z   in [2.5,   6.0]    (ultimate load factor)
      9. w_dg  in [1700,  2500]   (design gross weight)
      10. w_pp in [0.025, 0.08]    (weight per unit area)

    Fidelity parameter (stored as the 11th input):
      0: High fidelity (HF)
      1: Low fidelity 1 (LF1)
      2: Low fidelity 2 (LF2)
      3: Low fidelity 3 (LF2)

    The HF model is given by:
      f0 = 0.036 * s_w^0.758 * w_fw^0.0035 * (A/(cos^2(Lambda_rad)))^0.6 *
           q^0.006 * lam^0.04 * (100*t_c/cos(Lambda_rad))^-0.3 *
           (N_z*w_dg)^0.49 + s_w*w_pp

    LF models use slightly altered exponents and additive biases.
    """

    dim = 7
    _num_fidelities = 1
    _bounds = [
        (150.0, 200.0),  # s_w
        (220.0, 300.0),  # w_fw
        (6.0, 10.0),  # A
        (-10.0, 10.0),  # Lambda_deg
        (16.0, 45.0),  # q
        (0.5, 1.0),  # lam
        # (0.08, 0.18),  # t_c
        # (2.5, 6.0),  # N_z
        # (1700.0, 2500.0),  # w_dg
        # (0.025, 0.08),  # w_pp
        (0, 3), # fidelity
    ]
    fidelities = [0, 1, 2, 3]
    _optimal_value = 123.25

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # Expect X of shape [..., 11]: first 10 are design variables, last is fidelity index.
        s_w = X[..., 0]
        w_fw = X[..., 1]
        A = X[..., 2]
        Lambda_deg = X[..., 3]
        Lambda_rad = Lambda_deg * math.pi / 180.0

        q = X[..., 4]
        lam = X[..., 5]
        t_c = 0.13 #X[..., 6]
        N_z = 4.5 #X[..., 7]
        w_dg = 2000 #X[..., 8]
        w_pp = 0.05 #X[..., 9]
        fidelity = X[..., -1]
        cos_val = torch.cos(Lambda_rad)
        y = torch.zeros_like(s_w)
        # High fidelity (fidelity == 0)
        mask = fidelity == 0
        if mask.any():
            hf = (
                0.036
                * (s_w**0.758)
                * (w_fw**0.0035)
                * ((A / (cos_val**2)) ** 0.6)
                * (q**0.006)
                * (lam**0.04)
                * ((100.0 * t_c / cos_val) ** (-0.3))
                * ((N_z * w_dg) ** 0.49)
                + s_w * w_pp
            )
            y[mask] = hf[mask]
        # Low fidelity 1 (fidelity == 1)
        mask = fidelity == 1
        if mask.any():
            lf1 = (
                0.036
                * (s_w**0.758)
                * (w_fw**0.0035)
                * ((A / (cos_val**2)) ** 0.6)
                * (q**0.006)
                * (lam**0.04)
                * ((100.0 * t_c / cos_val) ** (-0.3))
                * ((N_z * w_dg) ** 0.49)
                + w_pp
            )
            y[mask] = lf1[mask]
        # Low fidelity 2 (fidelity == 2)
        mask = fidelity == 2
        if mask.any():
            lf2 = (
                0.036
                * (s_w**0.8)
                * (w_fw**0.0035)
                * ((A / (cos_val**2)) ** 0.6)
                * (q**0.006)
                * (lam**0.04)
                * ((100.0 * t_c / cos_val) ** (-0.3))
                * ((N_z * w_dg) ** 0.49)
                + w_pp
            )
            y[mask] = lf2[mask]
        # Low fidelity 3 (fidelity == 3)
        mask = fidelity == 3
        if mask.any():
            lf3 = (
                0.036
                * (s_w**0.9)
                * (w_fw**0.0035)
                * ((A / (cos_val**2)) ** 0.6)
                * (q**0.006)
                * (lam**0.04)
                * ((100.0 * t_c / cos_val) ** (-0.3))
                * ((N_z * w_dg) ** 0.49)
            )
            y[mask] = lf3[mask]
        return y

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        fidelity = X[..., 10]
        c = torch.zeros_like(fidelity)
        c[fidelity == 0] = 1000.0
        c[fidelity == 1] = 100.0
        c[fidelity == 2] = 10.0
        c[fidelity == 3] = 1.0
        return c


class BoreholeMultiFidelitySmall(SyntheticTestFunction):
    """Borehole Problem from [Chen2024]_.

    This problem models water flow through a borehole with 8 design variables:
          1. r_w   in [0.05,   0.15]   (borehole radius)
          2. r     in [100,    50000]  (radius of influence)
          3. T_u   in [63070,  115600] (transmissivity of upper aquifer)
          4. T_l   in [63.1,   116]    (transmissivity of lower aquifer)
          5. H_u   in [990,    1110]   (potentiometric head of upper aquifer)
          6. H_l   in [700,    820]    (potentiometric head of lower aquifer)
          7. L     in [1120,   1680]   (length of borehole)
          8. K_w   in [9855,   12045]  (hydraulic conductivity)

        The fidelity index (9th input) is categorical:
          0: High fidelity (HF)
          1: Low fidelity 1 (LF1)
          2: Low fidelity 2 (LF2)
          3: Low fidelity 3 (LF3)
          4: Low fidelity 4 (LF4)

        The HF model is defined by:
          f0 = (2*pi*T_u*(H_u-H_l)) / [ ln(r/r_w) * (1 + (2*L*T_l)/(ln(r/r_w)*r_w^2*K_w)) ]

        The low-fidelity models modify exponents and add a bias.
    """

    dim = 7
    _num_fidelities = 1
    _bounds = [
        (0.05, 0.15),  # r_w
        (100.0, 10000.0),  # r
        (100.0, 1000.0),  # T_u
        (10.0, 500.0),  # T_l
        (990.0, 1110.0),  # H_u
        (700.0, 820.0),  # H_l
        # (1000.0, 2000.0),  # L
        # (6000.0, 12000.0),  # K_w
        (0, 4),  # fidelity
    ]
    fidelities = [0, 1, 2, 3, 4]
    _optimal_value = 3.98

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r_w = X[..., 0]
        r = X[..., 1]
        T_u = X[..., 2]
        T_l = X[..., 3]
        H_u = X[..., 4]
        H_l = X[..., 5]
        L = 760 # X[..., 6]
        K_w = 1500 #X[..., 7]
        fidelity = X[..., -1]

        log_term = torch.log(r / r_w)
        numer = 2.0 * math.pi * T_u * (H_u - H_l)
        y = torch.zeros_like(r_w)

        # HF (fidelity 0)
        mask = fidelity == 0
        if mask.any():
            hf_denom = log_term * (
                1.0 + (2.0 * L * T_u) / (log_term * (r_w**2) * K_w) + T_u / T_l
            )
            hf = numer / hf_denom
            y[mask] = hf[mask]

        # LF1 (fidelity 1): add bias.
        mask = fidelity == 1
        if mask.any():
            lf1_numer = 2.0 * math.pi * T_u * (H_u - 0.8 * H_l)
            lf1_denom = log_term * (
                1.0 + (L * T_u) / (log_term * (r_w**2) * K_w) + T_u / T_l
            )
            lf1 = lf1_numer / lf1_denom
            y[mask] = lf1[mask]

        # LF2 (fidelity 2): modify the exponent on log_term and add bias.
        mask = fidelity == 2
        if mask.any():
            lf2_denom = log_term * (
                1.0 + (8 * L * T_u) / (log_term * (r_w**2) * K_w) + 0.75 * T_u / T_l
            )
            lf2 = numer / lf2_denom
            y[mask] = lf2[mask]

        # LF3 (fidelity 3): modify r_w exponent slightly.
        mask = fidelity == 3
        if mask.any():
            lf3_log_term = torch.log(4 * r / r_w)
            lf3_numer = 2.0 * math.pi * T_u * (1.09 * H_u - H_l)
            lf3_denom = lf3_log_term * (
                1.0 + (3 * L * T_u) / (log_term * (r_w**2) * K_w) + T_u / T_l
            )
            lf3 = lf3_numer / lf3_denom
            y[mask] = lf3[mask]
        # LF4 (fidelity 4): further bias.
        mask = fidelity == 4
        if mask.any():
            lf4_log_term = torch.log(2 * r / r_w)
            lf4_numer = 2.0 * math.pi * T_u * (1.05 * H_u - H_l)
            lf4_denom = lf4_log_term * (
                1.0 + (3 * L * T_u) / (log_term * (r_w**2) * K_w) + T_u / T_l
            )
            lf4 = lf4_numer / lf4_denom
            y[mask] = lf4[mask]

        return y

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        fidelity = X[..., 8]
        c = torch.zeros_like(fidelity)
        c[fidelity == 0] = 1000.0
        c[fidelity == 1] = 100.0
        c[fidelity == 2] = 10.0
        c[fidelity == 3] = 100.0
        c[fidelity == 4] = 10.0
        return c
    


def eval_pfn_on_problem(pfn, problem, train_X, train_Y, test_X, test_Y):
    task_id = train_X[:, -1].long()
    # normalize
    train_X = normalize(train_X, bounds=problem.bounds)
    test_X = normalize(test_X, bounds=problem.bounds)
    
    train_X = train_X[:, :-1]  # remove fidelity column
    test_X = test_X[:, :-1]  # remove fidelity column
    
    # standardize Y
    train_y_mean, train_y_std = train_Y.mean(), train_Y.std()
    train_Y = (train_Y - train_y_mean) / train_y_std
    
    with torch.no_grad():
        posterior = pfn_predict(pfn, task_id, train_X.float(), train_Y.float(), test_X.float())
        # compute sum of LL of each point in test set (using only marginal variances)
        var = posterior.variance
        mean = posterior.mean
        # import pdb; pdb.set_trace()
        unstand_mean, unstand_var = (
            mean * train_y_std + train_y_mean,
            var * train_y_std**2,
        )
        nll = (
            -Normal(loc=unstand_mean.squeeze(-1), scale=unstand_var.squeeze(-1))
            .log_prob(test_Y.view(-1))
            .sum(dim=-1)
            .mean()  # take average over MCMC samples (if needed)
            .item()
        )
        mse = (unstand_mean - test_Y).pow(2).mean().item()
    return nll, mse

# print(pickle.load(open("wing_weight_res.pkl", "rb")))
# print(pickle.load(open("borehole_res.pkl", "rb")))  # load the results
# lily

tkwargs = {"dtype": torch.float}
problem = WingWeightMultiFidelitySmall()
torch.manual_seed(0)
# define training and test set
N_TEST = 100
fidelity_to_n = {0: 5, 1: 5, 2: 10, 3: 50}
total_n = sum(fidelity_to_n.values())
train_X = torch.rand(total_n, problem.dim, **tkwargs)
train_X = unnormalize(train_X, bounds=problem.bounds)
# set fidelities
start = 0
for fidelity, n in fidelity_to_n.items():
    end = start + n
    train_X[start:end, -1] = fidelity
    start = end

train_Y = problem(train_X).unsqueeze(-1)

test_X = torch.rand(N_TEST, problem.dim, **tkwargs)
test_X = unnormalize(test_X, bounds=problem.bounds)
test_X[:, -1] = 0  # target fidelity
test_Y = problem(test_X).unsqueeze(-1)


wing_weight_res = eval_models_on_problem(problem, train_X, train_Y, test_X, test_Y)

# for pfn_id in ["vibrant-breeze-498", "revived-frog-499"]:
#     pfn = load_model(f"/home/yl9959/mtpfn/final_models/{pfn_id}")
#     nll, mse = eval_pfn_on_problem(pfn, problem, train_X, train_Y, test_X, test_Y)
#     wing_weight_res[pfn_id] = {"NLL": nll, "MSE": mse}
    
print("Wing Weight Results")
for k, v in wing_weight_res.items():
    print(k)
    print(v)
    print()
# Save the results
pickle.dump(wing_weight_res, open("wing_weight_res.pkl", "wb"))



problem = BoreholeMultiFidelitySmall()
torch.manual_seed(0)
# define training and test set
N_TEST = 100
fidelity_to_n = {0: 5, 1: 5, 2: 25, 3: 5, 4: 25}
total_n = sum(fidelity_to_n.values())
train_X = torch.rand(total_n, problem.dim, **tkwargs)
train_X = unnormalize(train_X, bounds=problem.bounds)
# set fidelities
start = 0
for fidelity, n in fidelity_to_n.items():
    end = start + n
    train_X[start:end, -1] = fidelity
    start = end

train_Y = problem(train_X).unsqueeze(-1)

test_X = torch.rand(N_TEST, problem.dim, **tkwargs)
test_X = unnormalize(test_X, bounds=problem.bounds)
test_X[:, -1] = 0  # target fidelity
test_Y = problem(test_X).unsqueeze(-1)

borehole_res = eval_models_on_problem(problem, train_X, train_Y, test_X, test_Y)

# for pfn_id in ["vibrant-breeze-498", "revived-frog-499"]:
#     pfn = load_model(f"/home/yl9959/mtpfn/final_models/{pfn_id}")
#     nll, mse = eval_pfn_on_problem(pfn, problem, train_X, train_Y, test_X, test_Y)
#     borehole_res[pfn_id] = {"NLL": nll, "MSE": mse}
    
print("Borehole Results")
for k, v in borehole_res.items():
    print(k)
    print(v)
    print()
# Save the results
pickle.dump(borehole_res, open("borehole_res.pkl", "wb"))