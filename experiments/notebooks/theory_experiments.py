# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# %matplotlib inline

# %%
import time
import itertools

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import root_scalar

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

# %% [markdown]
# ## DP-SGD

# %%
sample_rate = 0.001
num_steps = 10_000
classical_delta = 1e-5
accountant = opacus_acct.prv.PRVAccountant

delta_error = 1e-6
eps_error = 1e-6

# %%
adv_vals = np.concatenate([
    np.logspace(np.log10(0.01), np.log10(0.05), 10),
    np.linspace(0.05, 0.25, 10)
])
results_adv_calibration = []


for adv_val in tqdm.tqdm(list(adv_vals)):
    print(f"{adv_val=}")

    # Standard calibration.
    classical_epsilon = riskcal.utils.get_epsilon_for_advantage(delta=classical_delta, adv=adv_val)
    print(f"epsilon={classical_epsilon}, delta={classical_delta}")
    classical_noise = riskcal.blackbox.find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=classical_epsilon,
        delta=classical_delta,
        mu_max=100.0,
        eps_error=eps_error,
    )
    print(f"{classical_noise=}")

    # Advantage calibration.
    best_noise = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant=accountant,
        advantage=adv_val,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
    )
    noise_ratio = classical_noise / best_noise

    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )
    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    epsilon_ratio = best_epsilon / classical_epsilon

    print(f"{noise_ratio=} {epsilon_ratio=}")
    results_adv_calibration.append(
        dict(
            adv=adv_val,
            classical_noise=classical_noise,
            best_noise=best_noise,
            noise_ratio=noise_ratio,
            classical_epsilon=classical_epsilon,
            best_epsilon=best_epsilon,
            epsilon_ratio=epsilon_ratio
        )
    )

# %%
(
    pd.DataFrame(results_adv_calibration)
    .loc[:, ["adv", "classical_epsilon", "best_epsilon"]]
)

# %%
plt.figure()

g = sns.lineplot(
    data=(
        pd.DataFrame(results_adv_calibration)
        .query("adv >= 0.01")
        .melt(id_vars=["adv"],
              value_vars=["classical_noise", "best_noise"],
        )
        .rename(
            columns={
                "variable": "Method"
            }
        )
        .replace({
            "classical_noise": "Standard calibration",
            "best_noise": "Advantage calibration\hspace{2em}",
        })
    ),
    x="adv",
    y="value",
    hue="Method",
    marker="o",
)

g.set_xlabel(r"Attack advantage, $\eta$")
g.set_ylabel(r"Noise scale, $\sigma$")
g.set_yscale("log")

# plt.savefig("../images/dpsgd_adv_calibration.pdf", bbox_inches='tight')
plt.savefig("../images/dpsgd_adv_calibration.pgf", bbox_inches='tight', format="pgf")

# %%
tpr_vals = np.linspace(0.05, 0.5, 10)
tnr_vals = np.array([0.9, 0.95, 0.99])
results_delta_calibration = []

for tpr, tnr in tqdm.tqdm(list(itertools.product(tpr_vals, tnr_vals))):
    fpr = 1 - tnr
    fnr = 1 - tpr
    print(f"{fpr=} {fnr=}")

    if fpr + fnr >= 1:
        continue

    # Classical.
    classical_epsilon = riskcal.utils.get_epsilon_for_err_rates(classical_delta, alpha=fpr, beta=fnr)
    classical_noise = riskcal.blackbox.find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=classical_epsilon,
        delta=classical_delta,
        mu_max=100.0,
        eps_error=eps_error,
    )
    print(f"(epsilon={classical_epsilon:.4f}, delta={classical_delta:.5f}): noise={classical_noise}")

    # Risk calibration for error rates.
    noise_multiplier = riskcal.pld.find_noise_multiplier_for_err_rates(
        alpha=fpr,
        beta=fnr,
        sample_rate=sample_rate,
        num_steps=num_steps,
    )

    best_noise = noise_multiplier
    noise_ratio = classical_noise / best_noise

    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )

    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    epsilon_ratio = best_epsilon / classical_epsilon

    print(f"{noise_ratio=} {epsilon_ratio=}")
    results_delta_calibration.append(
        dict(
            tpr=tpr,
            tnr=tnr,
            fnr=fnr,
            fpr=fpr,
            classical_noise=classical_noise,
            best_noise=best_noise,
            noise_ratio=noise_ratio,
            classical_epsilon=classical_epsilon,
            best_epsilon=best_epsilon,
            epsilon_ratio=epsilon_ratio,
        )
    )

# %%
(
    pd.DataFrame(results_delta_calibration)
)

# %%
plt.figure()

sns.relplot(
    data=(
        pd.DataFrame(results_delta_calibration)
        .melt(
            id_vars=["tpr", "fpr"],
            value_vars=["classical_noise", "best_noise"],
        )
        .assign(fpr=lambda df: df.fpr.round(3))
        .replace({
            "classical_noise": "Standard calibration",
            "best_noise": "TPR/FPR calibration",
        })
        .rename(
            columns={
                "fpr": r"Attack FPR, $\alpha$",
                "tpr": r"Attack TPR, $1 - \beta$",
                "value": "Noise scale, $\sigma$",
                "variable": "Method"
            }
        )
    ),
    x=r"Attack TPR, $1 - \beta$",
    y="Noise scale, $\sigma$",
    col=r"Attack FPR, $\alpha$",
    hue="Method",
    marker="o",
    kind="line",
    # facet_kws={'sharey': False, 'sharex': True},
)

plt.xlim(0.05, 0.55)

# plt.savefig("../images/dpsgd_err_rates_calibration.pdf", bbox_inches='tight')
plt.savefig("../images/dpsgd_err_rates_calibration.pgf", bbox_inches='tight', format="pgf")

# %%
# %%timeit
riskcal.pld.get_beta(alpha=0.01, noise_multiplier=1.0, sample_rate=0.001, num_steps=10_000)

# %%
# %%timeit
riskcal.pld.find_noise_multiplier_for_err_rates(alpha=0.01, beta=0.2, sample_rate=0.001, num_steps=10_000)
