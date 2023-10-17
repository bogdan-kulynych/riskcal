# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import root_scalar

from tqdm import autonotebook as tqdm

from opacus import accountants as opacus_acct

from matplotlib import pyplot as plt

sns.set(style="whitegrid", context="paper", font_scale=1.75)

# %%
import riskcal


# %% [markdown]
# ## Gaussian mechanism

# %%
class GaussianMechanismAccountant:
    def __init__(self):
        pass

    def step(self, noise_multiplier: float, **kwargs):
        self.noise_multiplier = noise_multiplier

    def get_epsilon(self, delta):
        def get_delta(epsilon):
            return norm.cdf(0.5/self.noise_multiplier - epsilon * self.noise_multiplier) \
                 - np.exp(epsilon) * norm.cdf(-0.5/self.noise_multiplier - epsilon * self.noise_multiplier)

        return root_scalar(lambda epsilon: get_delta(epsilon) - delta, x0 = 0.0).root


accountant = GaussianMechanismAccountant
sample_rate = 1
num_steps = 1

# %%
adv_vals = np.linspace(0.1, 0.5, 20)
results_gm_adv_calibration = []


for adv_val in tqdm.tqdm(list(adv_vals)):
    print(f"{adv_val=}")
    
    # Standard calibration.
    classical_epsilon = riskcal.utils.get_epsilon_for_advantage(delta=classical_delta, adv=adv_val)
    print(f"epsilon={classical_epsilon}, delta={classical_delta}")
    classical_noise = riskcal.find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=classical_epsilon,
        delta=classical_delta,
        mu_max=100.0,
        eps_error=eps_error,
    )
    print(f"{classical_noise=}")
    results_gm_adv_calibration.append(
        dict(
            adv=adv_val,
            noise=classical_noise,
            epsilon=classical_epsilon,
            method="standard",
        )
    )

    # Generic advantage calibration.
    best_noise = riskcal.find_noise_multiplier_for_advantage(
        accountant=accountant,
        advantage=adv_val,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=0.1,
    )
    
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )
    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    
    results_gm_adv_calibration.append(
        dict(
            adv=adv_val,
            noise=best_noise,
            epsilon=best_epsilon,
            method="generic",
        )
    )

    # Specialized advantage calibration.
    best_noise = 1 / (2 * norm.ppf((adv_val + 1) / 2))
    
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )
    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    
    results_gm_adv_calibration.append(
        dict(
            adv=adv_val,
            noise=best_noise,
            epsilon=best_epsilon,
            method="gaussian",
        )
    )

# %%
results_gm_adv_calibration

# %%
g = sns.lineplot(
    data=(
        pd.DataFrame(results_gm_adv_calibration)
        .rename(
            columns={
                "method": "Method"
            }
        )
        .replace({
            "standard": "Standard CF calibration",
            "generic": "Mechanism-agnostic adv. calibration",
            "gaussian": "Specialized adv. calibration",
        })
    ),
    x="adv",
    y="noise",
    hue="Method",
)

g.set_xlabel("Adversary's advantage")
g.set_ylabel("Noise scale")

plt.savefig("../images/gm_adv_calibration.pdf", bbox_inches='tight')

# %%
delta_error = 1e-5
eps_error = 1e-7

tpr_vals = np.linspace(0.1, 0.5, 10)
tnr_vals = np.array([0.9, 0.95, 0.99])
results_gm_delta_calibration = []

for tpr, tnr in tqdm.tqdm(list(itertools.product(tpr_vals, tnr_vals))):
    fpr = 1 - tnr
    fnr = 1 - tpr
    print(f"{fpr=} {fnr=}")
    
    if fpr + fnr >= 1:
        continue

    # Classical.       
    classical_epsilon = riskcal.utils.get_epsilon_for_err_rates(classical_delta, alpha=fpr, beta=fnr)    
    classical_noise = riskcal.find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=classical_epsilon,
        delta=classical_delta,
        mu_max=100.0,
        eps_error=eps_error,
    )
    results_gm_delta_calibration.append(
        dict(
            tpr=tpr,
            tnr=tnr,
            fnr=fnr,
            fpr=fpr,
            noise=classical_noise,
            epsilon=classical_epsilon,
            internal_epsilon=classical_epsilon,
            internal_delta=classical_delta,
            method="standard",
        )
    )

    # Generic risk calibration for error rates.
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant=accountant,
        alpha=fpr,
        beta=fnr,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        eps_error=eps_error,
    )

    best_noise = calibration_result.noise_multiplier
    internal_delta = calibration_result.calibration_delta
    internal_epsilon = calibration_result.calibration_epsilon
    
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )

    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    results_gm_delta_calibration.append(
        dict(
            tpr=tpr,
            tnr=tnr,
            fnr=fnr,
            fpr=fpr,
            noise=best_noise,
            epsilon=best_epsilon,
            internal_epsilon=internal_epsilon,
            internal_delta=internal_delta,
            method="generic"
        )
    )

    # Generic risk calibration for error rates.
    best_noise = np.sqrt(1 / ( norm.ppf(1-fpr) - norm.ppf(fnr) ))
    
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(
            noise_multiplier=best_noise,
            sample_rate=sample_rate,
        )

    best_epsilon = acct_obj.get_epsilon(delta=classical_delta)
    results_gm_delta_calibration.append(
        dict(
            tpr=tpr,
            tnr=tnr,
            fnr=fnr,
            fpr=fpr,
            noise=best_noise,
            epsilon=best_epsilon,
            method="gaussian"
        )
    )

# %%
sns.relplot(
    data=(
        pd.DataFrame(results_gm_delta_calibration)
        .assign(fpr=lambda df: df.fpr.round(2))
        .replace({
            "standard": "Standard CF calibration",
            "generic": "Mechanism-agnostic TPR/FPR calibration",
            "gaussian": "Specialized TPR/FPR calibration",
        })
        .rename(
            columns={
                "fpr": "FPR",
                "tpr": "TPR (attack sensitivity)",
                "noise": "Noise scale",
                "method": "Method",
            }
        )
    ),
    x="TPR (attack sensitivity)",
    y="Noise scale",
    col="FPR",
    hue="Method",
    kind="line",
    # facet_kws={'sharey': False, 'sharex': True},
)

plt.savefig("../images/gm_err_rates_calibration.pdf", bbox_inches='tight')

# %% [markdown]
# ## DP-SGD

# %%
sample_rate = 0.001
num_steps = 10_000
classical_delta = 1e-5
accountant = opacus_acct.rdp.RDPAccountant

delta_error = 0.05
eps_error = 0.001

# %%
adv_vals = np.linspace(0.1, 0.5, 20)
results_adv_calibration = []


for adv_val in tqdm.tqdm(list(adv_vals)):
    print(f"{adv_val=}")
    
    # Standard calibration.        
    classical_epsilon = riskcal.utils.get_epsilon_for_advantage(delta=classical_delta, adv=adv_val)
    print(f"epsilon={classical_epsilon}, delta={classical_delta}")
    classical_noise = riskcal.find_noise_multiplier_for_epsilon_delta(
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
    best_noise = riskcal.find_noise_multiplier_for_advantage(
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
g = sns.lineplot(
    data=(
        pd.DataFrame(results_adv_calibration)
        .melt(id_vars=["adv"],
              value_vars=["classical_noise", "best_noise"],
        )
        .rename(
            columns={
                "variable": "Method"
            }
        )
        .replace({
            "classical_noise": "Standard CF calibration",
            "best_noise": "Advantage calibration",
        })
    ),
    x="adv",
    y="value",
    hue="Method",
)

g.set_xlabel("Adversary's advantage")
g.set_ylabel("Noise scale")

plt.savefig("../images/dpsgd_adv_calibration.pdf", bbox_inches='tight')

# %%
delta_error = 1e-5

tpr_vals = np.linspace(0.1, 0.5, 10)
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
    classical_noise = riskcal.find_noise_multiplier_for_epsilon_delta(
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
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant=accountant,
        alpha=fpr,
        beta=fnr,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        eps_error=eps_error,
    )

    best_noise = calibration_result.noise_multiplier
    internal_delta = calibration_result.calibration_delta
    internal_epsilon = calibration_result.calibration_epsilon
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
            internal_epsilon=internal_epsilon,
            internal_delta=internal_delta,
        )
    )

# %%
(
    pd.DataFrame(results_delta_calibration)
)

# %%
sns.relplot(
    data=(
        pd.DataFrame(results_delta_calibration)
        .melt(
            id_vars=["tpr", "fpr"],
            value_vars=["classical_noise", "best_noise"],
        )
        .assign(fpr=lambda df: df.fpr.round(2))
        .replace({
            "classical_noise": "Standard CF calibration",
            "best_noise": "TPR/FPR calibration",

        })
        .rename(
            columns={
                "fpr": "FPR",
                "tpr": "TPR (attack sensitivity)",
                "value": "Noise scale",
                "variable": "Method"
            }
        )
    ),
    x="TPR (attack sensitivity)",
    y="Noise scale",
    col="FPR",
    hue="Method",
    kind="line",
    # facet_kws={'sharey': False, 'sharex': True},
)

plt.savefig("../images/dpsgd_err_rates_calibration.pdf", bbox_inches='tight')
