# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
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
# -

import riskcal

exp_metadata = pd.read_csv("../data/cifar10_metadata.csv", index_col=0)

exp_metadata

# +
# Reimplemented based on https://github.com/ftramer/Handcrafted-DP/blob/main/dp_utils.py
from dp_accounting.pld import privacy_loss_distribution

def get_pld(noise_multiplier, epochs, batch_size=8192, data_size=50000, grid_size=1e-4):
    norm_noise_multiplier = 8.0
    sgd_noise_multiplier = noise_multiplier
    sample_rate = batch_size / data_size
    steps = int(epochs * np.ceil(data_size / batch_size))
    
    # from https://github.com/ftramer/Handcrafted-DP/blob/main/cnns.py
    # compute the budget spent in normalization.
    pld_norm = (privacy_loss_distribution
        .from_gaussian_mechanism(standard_deviation=norm_noise_multiplier,
                                 use_connect_dots=True,
                                 value_discretization_interval=grid_size)
        .self_compose(2)
    )
    
    pld_sgd = (privacy_loss_distribution
        .from_gaussian_mechanism(standard_deviation=sgd_noise_multiplier,
                                 sampling_prob=sample_rate,
                                 use_connect_dots=True,
                                 value_discretization_interval=grid_size)
        .self_compose(steps)
    )

    return pld_norm.compose(pld_sgd)

def get_epsilon_from_pld(pld, delta=1e-5):
    return pld.get_epsilon_for_delta(delta)

def get_beta(pld, alpha=0.1):
    return np.minimum(
        riskcal.pld.get_beta_from_pld(pld, alpha=alpha),
        riskcal.pld.get_alpha_from_pld(pld, beta=alpha),
    )


# +
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_renyi_divergence(sample_rate, noise_multiplier, orders=ORDERS):
    rdp = opacus_acct.analysis.rdp.compute_rdp(
        q=sample_rate, noise_multiplier=noise_multiplier, steps=1, orders=orders
    )
    return rdp

def get_epsilon_from_rdp(rdp, delta=1e-5, orders=ORDERS):
    return opacus_acct.analysis.rdp.get_privacy_spent(
        rdp=rdp, delta=delta, orders=orders
    )[0]

def get_grid_search_beta(rdp, alpha=0.1, grid_size=1000, orders=ORDERS):
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


# -

for i, row in exp_metadata.iterrows():
    eps = get_epsilon_from_pld(get_pld(noise_multiplier=row.sigma, batch_size=row.batch_size, epochs=row.epochs), delta=1e-5)
    print(f"sigma={row.sigma:.2f}, eps={eps:.4f}")

# +
standard_delta = 1e-5
alphas = np.array([0.01, 0.05, 0.1])

plot_data = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    pld = get_pld(
        noise_multiplier=row.sigma, epochs=row.epochs, batch_size=row.batch_size
    )
    rdp = get_rdp(
        noise_multiplier=row.sigma, epochs=row.epochs, batch_size=row.batch_size
    )

    # CF delta
    standard_pld_eps = get_epsilon_from_pld(pld, standard_delta)
    standard_rdp_eps = get_epsilon_from_rdp(rdp, standard_delta)

    for alpha in alphas:
        standard_pld_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            standard_pld_eps, standard_delta, alpha=alpha
        )
        standard_rdp_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            standard_rdp_eps, standard_delta, alpha=alpha
        )
    
        # FPR/FNR calibrated (tight accounting)
        cal_beta = get_beta(pld, alpha=alpha)
    
        plot_data.append(
            dict(
                alpha=alpha,
                standard_pld_beta=standard_pld_beta,
                standard_pld_tpr=1 - standard_pld_beta,
                standard_rdp_beta=standard_rdp_beta,
                standard_rdp_tpr=1 - standard_rdp_beta,
                cal_beta=cal_beta,
                cal_tpr=1 - cal_beta,
                test_acc=row.test_acc,
                sigma=row.sigma,
            )
        )

# +
g = sns.relplot(
    data=(
        pd.DataFrame(plot_data)
        .melt(
            id_vars=["alpha", "test_acc", "sigma"],
            value_vars=["cal_tpr", "standard_rdp_tpr", "standard_pld_tpr"],
            var_name="Method",
            value_name="Attack risk",
        )
        .replace(
            {"cal_tpr": "Attack risk calibration",
             "standard_rdp_tpr": "Standard calibration (RDP analysis)",
             "standard_pld_tpr": "Standard calibration (tight re-analysis)"}
        )
        .rename(
            columns={
                "alpha": r"$\alpha$",
                "test_acc": "Accuracy",
                "sigma": "Noise scale",
            }
        )
    ),
    x="Accuracy",
    y="Attack risk",
    hue="Method",
    hue_order=["Standard calibration (tight re-analysis)", "Attack risk calibration", "Standard calibration (RDP analysis)"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

for item, ax in g.axes_dict.items():
    ax.set_title("")
# +
g = sns.relplot(
    data=(
        pd.DataFrame(plot_data)
        .melt(
            id_vars=["alpha", "test_acc", "sigma"],
            value_vars=["cal_tpr", "standard_pld_tpr"],
            var_name="Method",
            value_name="Attack risk",
        )
        .replace(
            {"cal_tpr": "Attack risk calibration",
             "standard_pld_tpr": "Standard calibration"}
        )
        .rename(
            columns={
                "alpha": r"$\alpha$",
                "test_acc": "Accuracy",
                "sigma": "Noise scale",
            }
        )
    ),
    x="Accuracy",
    y="Attack risk",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

for item, ax in g.axes_dict.items():
    ax.set_title("")

plt.savefig("../images/cifar10_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")
