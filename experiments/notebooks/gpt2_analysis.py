# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
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

exp_metadata = pd.read_csv("../data/gpt2_metadata.csv", index_col=None)

exp_metadata = (
    exp_metadata
    .query("sigma < 2")
    .query("eps not in [3.0, 3.75, 6.00]")
    .sort_values(by="sigma")
)
exp_metadata

0.705275 - 0.521789


# +
# Re-implementation with PRV accounting.
def get_epsilon(noise_multiplier, sample_rate, num_steps, delta):
    acct = opacus_acct.prv.PRVAccountant()
    for _ in range(int(num_steps)):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acct.get_epsilon(delta)
    

def get_beta(noise_multiplier, sample_rate, num_steps, alpha, discretization=0.0001):
    return riskcal.pld.get_beta(
        alpha=alpha,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        num_steps=int(num_steps)
    )


# +
cf_delta = 1e-5
alphas = [0.01, 0.05, 0.1]

plot_data = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    # CF delta
    cf_eps = get_epsilon(row.sigma, row.q, row.steps, cf_delta)

    for alpha in alphas:
        cf_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            cf_eps, cf_delta, alpha=alpha
        )

        # FPR/FNR calibrated
        cal_beta = get_beta(row.sigma, row.q, row.steps, alpha=alpha)    
        plot_data.append(
            dict(
                alpha=alpha,
                cf_beta=cf_beta,
                cf_tpr=1 - cf_beta,
                cf_eps=cf_eps,
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
        .assign(test_acc=lambda df: df.test_acc * 100)
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

plt.xlim(0, 1.0)

plt.savefig("../images/gpt2_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")

# +
g = sns.relplot(
    data=(
        pd.DataFrame(plot_data)
        .assign(test_acc=lambda df: df.test_acc * 100)
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
    y="Attack risk",
    x="Accuracy",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

# plt.savefig("../images/gpt2_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")
# -

for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    eps = get_epsilon(row.sigma, row.q, row.steps, cf_delta)
    naive_adv = riskcal.utils.get_err_rate_for_epsilon_delta(alpha=0.1, epsilon=eps, delta=cf_delta)
    print(naive_adv)


