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

exp_metadata = pd.read_csv("../data/gpt2_metadata.csv", index_col=0)

exp_metadata = exp_metadata.sort_values(by="sigma")
exp_metadata

# Accuracy difference
exp_metadata.test_acc.max() - exp_metadata.test_acc.min()

# +
standard_delta = 1e-5

for i, row in exp_metadata.iterrows():
    acct = riskcal.dpsgd.CTDAccountant()
    for _ in range(int(row.steps)):
        acct.step(noise_multiplier=row.sigma, sample_rate=row.q)
    print(acct.get_epsilon(delta=standard_delta))

# +
alpha = np.array([0.01, 0.05, 0.1])

plot_chunks = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    acct = riskcal.dpsgd.CTDAccountant()
    for _ in range(int(row.steps)):
        acct.step(noise_multiplier=row.sigma, sample_rate=row.q)
        
    standard_eps = acct.get_epsilon(delta=standard_delta)
    standard_beta = riskcal.conversions.get_beta_for_epsilon_delta(
        standard_eps, standard_delta, alpha=alpha
    )
    cal_beta = acct.get_beta(alpha=alpha)

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
y_label = r"Attack risk (TPR, $1 - \beta$)"
g = sns.relplot(
    data=(
        pd.concat(plot_chunks, axis=0)
        .assign(test_acc=lambda df: df.test_acc * 100)
        .melt(
            id_vars=["alpha", "test_acc", "sigma"],
            value_vars=["cal_tpr", "standard_tpr"],
            var_name="Method",
            value_name=y_label,
        )
        .replace(
            {"cal_tpr": "Attack risk calibration", "standard_tpr": "Standard calibration"}
        )
        .rename(
            columns={
                "alpha": r"$\alpha$",
                "test_acc": "Task accuracy",
                "sigma": "Noise scale",
            }
        )
    ),
    y=y_label,
    x="Task accuracy",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

plt.ylim(0, 1.0)

g.axes[0, 0].set_xlabel("")
g.axes[0, 2].set_xlabel("")

plt.savefig(
    "../images/gpt2_err_rates_calibration.pgf", bbox_inches="tight", format="pgf"
)

# +
g = sns.relplot(
    data=(
        pd.concat(plot_chunks, axis=0)
        .assign(test_acc=lambda df: df.test_acc * 100)
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
    x="Attack risk",
    y="Accuracy",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

plt.savefig(
    "../images/gpt2_err_rates_calibration_flipped.pgf",
    bbox_inches="tight",
    format="pgf",
)

# +
alpha = np.linspace(0, 1, 200)
plot_chunks = []

for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    acct = riskcal.dpsgd.CTDAccountant()
    for _ in range(int(row.steps)):
        acct.step(noise_multiplier=row.sigma, sample_rate=row.q)

    pld_beta = acct.get_beta(alpha=alpha)
    dp_beta = riskcal.conversions.get_beta_for_epsilon_delta(acct.get_epsilon(delta=standard_delta), standard_delta, alpha)
    plot_chunks.extend(
        [
            pd.DataFrame(dict(
                alpha=alpha,
                beta=dp_beta,
                acc=row.test_acc,
                sigma=row.sigma,
                method=r"$(\varepsilon, \delta)$-DP ",
            )),
            pd.DataFrame(dict(
                alpha=alpha,
                beta=pld_beta,
                acc=row.test_acc,
                sigma=row.sigma,
                method="Ours (Algorithm 1)",
            ))
        ]
    )

# +
sns.relplot(
    data=(
        pd.concat(plot_chunks, axis=0)
        .query("0.55 < acc < 0.705")
        .assign(acc=lambda df: df.acc.round(2))
        .rename(columns={
            "acc": "Task accuracy",
            "alpha": r"Attack FPR, $\alpha$",
            "beta": r"Attack FNR, $\beta$",
            "method": "Method",
        })
    ),
    x= r"Attack FPR, $\alpha$",
    y= r"Attack FNR, $\beta$",
    hue="Method",
    col="Task accuracy",
    kind="line",
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.savefig(
    "../images/gpt2_trade_off_curves.pgf",
    bbox_inches="tight",
    format="pgf",
)
