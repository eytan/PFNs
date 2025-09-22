
from gen_batch import (
    combine_batch,
    get_hpo_batch_fn,
    get_icm,
    get_lmc,
    get_mtgp_batch,
    get_pd1_surrogate_batch_fn,
    get_pd1_eval_batch_fn,
    toy_no_memory_batch,
    get_trios_batch,
)

from PFNs.pfns.priors.hebo_prior import get_batch as get_hebo_batch
import matplotlib.pyplot as plt


mtgp_batch = get_mtgp_batch(
    batch_size=200,
    seq_len=50,
    num_features=1,
    max_num_tasks=2,
    num_tasks=1,
    lengthscale=None,
    hyperparameters={},
    device="cpu"
)
hebo_batch = get_hebo_batch(
    batch_size=200,
    seq_len=50,
    num_features=1
)
plt.hist(mtgp_batch.x.flatten(), density=True, alpha=0.5, label="mtgp")
plt.hist(hebo_batch.x.cpu().flatten(), density=True, alpha=0.5, label="hebo")
plt.legend()
plt.savefig("1x.png")
plt.close()
plt.hist(mtgp_batch.y.flatten(), density=True, alpha=0.5, label="mtgp")
plt.hist(hebo_batch.y.cpu().flatten(), density=True, alpha=0.5, label="hebo")
plt.legend()
plt.savefig("1y.png")


# def get_batch(
#     batch_size,
#     seq_len,
#     num_features,
#     device=default_device,
#     hyperparameters=None,
#     batch_size_per_gp_sample=None,
#     single_eval_pos=None,
#     fix_to_range=None,
#     equidistant_x=False,
#     verbose=False,
#     **kwargs,


# plt.savefig("1hebo.png")
# plt.close()
# plt.savefig("1heboy.png")