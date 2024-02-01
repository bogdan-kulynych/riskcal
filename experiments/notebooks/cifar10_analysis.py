# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import itertools

import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from tqdm import autonotebook as tqdm

from opacus import accountants as opacus_acct

import matplotlib as mpl
from matplotlib import pyplot as plt

from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

sns.set(
    style="whitegrid", context="paper", font_scale=2,
    rc={"lines.linewidth": 2.5, "lines.markersize": 6, "lines.markeredgewidth": 0.0}
)
plt.rcParams.update({
    "font.family": "sans-serif",  # use serif/main font for text elements
    "font.serif": "Helvetica",
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

# %%
import riskcal

# %%
exp_metadata = pd.read_csv("../data/experiment_metadata.csv", index_col=0)

# %%
exp_metadata

# %%
# Based on https://github.com/ftramer/Handcrafted-DP/blob/main/dp_utils.py
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_renyi_divergence(sample_rate, noise_multiplier, orders=ORDERS):
    rdp = opacus_acct.analysis.rdp.compute_rdp(
        q=sample_rate, noise_multiplier=noise_multiplier, steps=1, orders=orders
    )
    return rdp


def get_privacy_spent(rdp, delta=1e-5, orders=ORDERS):
    return opacus_acct.analysis.rdp.get_privacy_spent(
        rdp=rdp, delta=delta, orders=orders
    )[0]


def get_beta(rdp, alpha=0.1, grid_size=1000, orders=ORDERS):
    deltas = np.linspace(0, 1 - alpha, grid_size)
    betas = []
    for delta in deltas:
        beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            get_privacy_spent(rdp, delta=delta, orders=orders),
            delta=delta,
            alpha=alpha,
        )
        betas.append(beta)

    best_index = np.argmax(betas)
    return betas[best_index], deltas[best_index]


# %%
def get_rdp(noise_multiplier, epochs, batch_size=8192, data_size=50000, orders=ORDERS):
    orders = ORDERS
    bn_noise_multiplier = 8.0
    sample_rate = batch_size / data_size
    steps = np.ceil(data_size / batch_size)
    noise_multiplier = noise_multiplier

    # from https://github.com/ftramer/Handcrafted-DP/blob/main/cnns.py
    # compute the RDP spent in normalization.
    rdp_norm = 2 * get_renyi_divergence(
        sample_rate=1.0, noise_multiplier=bn_noise_multiplier, orders=orders
    )

    rdp_sgd = (
        get_renyi_divergence(sample_rate, noise_multiplier, orders=orders)
        * steps
        * epochs
    )
    return rdp_sgd + rdp_norm


# %%
cf_delta = 1e-5
alphas = [0.01, 0.05, 0.1]

plot_data = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    rdp = get_rdp(
        noise_multiplier=row.sigma, epochs=row.epochs, batch_size=row.batch_size
    )

    # CF delta
    cf_eps = get_privacy_spent(rdp, cf_delta)

    for alpha in alphas:
        cf_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            cf_eps, cf_delta, alpha=alpha
        )

        # FPR/FNR calibrated
        cal_beta, cal_delta = get_beta(rdp, alpha=alpha)
        cal_eps = get_privacy_spent(rdp, cal_delta)

        plot_data.append(
            dict(
                alpha=alpha,
                cf_beta=cf_beta,
                cf_tpr=1 - cf_beta,
                cf_eps=cf_eps,
                cal_eps=cal_eps,
                cal_delta=cal_delta,
                cal_beta=cal_beta,
                cal_tpr=1 - cal_beta,
                test_acc=row.test_acc,
                sigma=row.sigma,
            )
        )

# %%
g = sns.relplot(
    data=(
        pd.DataFrame(plot_data)
        .melt(
            id_vars=["alpha", "test_acc", "sigma"],
            value_vars=["cal_tpr", "cf_tpr"],
            var_name="Method",
            value_name="Attack risk",
        )
        .replace(
            {"cal_tpr": "Attack risk calibration", "cf_tpr": "Standard calibration"}
        )
        .rename(
            columns={
                "alpha": r"$\alpha$",
                "test_acc": "Accuracy",
                "sigma": "Noise scale",
            }
        )
    ),
    y="Accuracy",
    x="Attack risk",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

plt.xlim(0, 1)
# plt.savefig("../images/cifar10_err_rates_calibration.pdf", bbox_inches="tight")
plt.savefig("../images/cifar10_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")

# %%
pd.DataFrame(plot_data)

# %%
