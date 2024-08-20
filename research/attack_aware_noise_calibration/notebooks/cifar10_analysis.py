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

mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)

sns.set(
    style="whitegrid",
    context="paper",
    font_scale=2,
    rc={"lines.linewidth": 2.5, "lines.markersize": 6, "lines.markeredgewidth": 0.0},
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",  # use serif/main font for text elements
        "font.serif": "Helvetica",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)
# -

import riskcal

exp_metadata = pd.read_csv("../data/cifar10_metadata.csv", index_col=0)

exp_metadata

# +
# Reimplemented based on https://github.com/ftramer/Handcrafted-DP/blob/main/dp_utils.py
from dp_accounting.pld import privacy_loss_distribution


def get_pld(noise_multiplier, epochs, batch_size=8192, data_size=50000):
    norm_noise_multiplier = 8.0
    sgd_noise_multiplier = noise_multiplier
    sample_rate = batch_size / data_size
    steps = int(epochs * np.ceil(data_size / batch_size))

    # from https://github.com/ftramer/Handcrafted-DP/blob/main/cnns.py
    # compute the budget spent in normalization.
    pld_norm = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=norm_noise_multiplier, use_connect_dots=True
    ).self_compose(2)

    pld_sgd = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sgd_noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
    ).self_compose(steps)

    return pld_norm.compose(pld_sgd)


def get_epsilon(pld, delta=1e-5):
    return pld.get_epsilon_for_delta(delta)


# +
standard_delta = 1e-5
alpha = np.array([0.01, 0.05, 0.1])

plot_chunks = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    pld = get_pld(
        noise_multiplier=row.sigma, epochs=row.epochs, batch_size=row.batch_size
    )
    rdp = get_rdp(
        noise_multiplier=row.sigma, epochs=row.epochs, batch_size=row.batch_size
    )

    standard_eps = pld.get_epsilon_for_delta(standard_delta)
    standard_beta = riskcal.conversions.get_beta_for_epsilon_delta(
        standard_eps, standard_delta, alpha=alpha
    )
    cal_beta = riskcal.get_beta_from_pld(pld, alpha=alpha)

    plot_chunks.append(
        pd.DataFrame(dict(
            alpha=alpha,
            standard_beta=standard_beta,
            standard_tpr=1 - standard_beta,
            standard_eps=standard_eps,
            cal_beta=cal_beta,
            cal_tpr=1 - cal_beta,
            test_acc=row.test_acc,
            sigma=row.sigma,
        ))
    )

# +
g = sns.relplot(
    data=(
        pd.concat(plot_chunks, axis=0)
        .melt(
            id_vars=["alpha", "test_acc", "sigma"],
            value_vars=["cal_tpr", "standard_tpr"],
            var_name="Method",
            value_name="Attack risk",
        )
        .replace(
            {"cal_tpr": "Attack risk calibration", "standard_tpr": "Standard calibration"}
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

# plt.savefig("../images/cifar10_err_rates_calibration.pdf", bbox_inches="tight")
plt.savefig(
    "../images/cifar10_err_rates_calibration.pgf", bbox_inches="tight", format="pgf"
)
# -


