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

exp_metadata = pd.read_csv("../data/gpt2_metadata.csv", index_col=0).drop(columns="exp")

exp_metadata

# Accuracy difference
exp_metadata.test_acc.max() - exp_metadata.test_acc.min()


# +
def get_epsilon(noise_multiplier, sample_rate, num_steps, delta):
    acct = riskcal.pld.CTDAccountant()
    for _ in range(int(num_steps)):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acct.get_epsilon(delta=delta)
    

def get_beta(noise_multiplier, sample_rate, num_steps, alpha):
    return np.minimum(
        riskcal.pld.get_beta(
            alpha,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            num_steps=int(num_steps)
        ),
        riskcal.pld.get_alpha(
            alpha,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            num_steps=int(num_steps)
        )
    )


# -

# Get epsilons under standard calibration.

for i, row in exp_metadata.iterrows():
    eps = get_epsilon(noise_multiplier=row.sigma, sample_rate=row.q, num_steps=row.steps, delta=1e-5)
    print(f"sigma={row.sigma:.2f}, eps={eps:.4f}")

# Plot trade-off curves

# +
import riskcal
import numpy as np
from matplotlib import pyplot as plt

delta = 1e-5
data_chunks = []

for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    noise_multiplier = row.sigma
    sample_rate = row.q
    num_steps = row.steps
    print(f"{noise_multiplier=} {sample_rate=} {num_steps=}")
    
    pld = riskcal.pld.privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=1e-4
    )
    pld = pld.self_compose(int(num_steps))
    epsilon = pld.get_epsilon_for_delta(delta)
    adv = pld.get_delta_for_epsilon(0)
    print(f"{epsilon=} {delta=}")
    print(f"adv={adv} acc={0.5 * (adv + 1)}")
    
    alphas = np.linspace(1e-5, 1 - 1e-5, 200)
    betas = np.minimum(
        riskcal.pld.get_beta_from_pld(pld, alphas),
        riskcal.pld.get_alpha_from_pld(pld, alphas),
    )
    
    hockeystick_betas = []
    for alpha in alphas:
        hockeystick_betas.append(
            riskcal.utils.get_err_rate_for_epsilon_delta(
                alpha=alpha,
                epsilon=epsilon,
                delta=delta,
            )
        )
    hockeystick_betas = np.array(hockeystick_betas)
    perfect_betas = 1 - alphas

    model_name = rf"{row.test_acc * 100:.2f}\% accuracy"
    data_chunks.extend([
        pd.DataFrame(dict(alpha=alphas, beta=betas, model=model_name, curve="exact")),
        pd.DataFrame(dict(alpha=alphas, beta=hockeystick_betas, model=model_name, curve="dp")),
        pd.DataFrame(dict(alpha=alphas, beta=perfect_betas, model=model_name, curve="perfect")),
    ])

# +
curve_label_dict = {
    "dp": r"$(\varepsilon, \delta)$-DP trade-off curve (used for `Standard Calibration')",
    "exact": r"Trade-off curve from Alg. 1 (used for `Attack Risk Calibration')",
    "perfect": "Perfect privacy",
}
order = [curve_label_dict["dp"], curve_label_dict["exact"], curve_label_dict["perfect"]]

g = sns.relplot(
    data=(
        pd.concat(data_chunks)
        .replace(curve_label_dict)
    ),
    x="alpha",
    y="beta",
    hue="curve",
    style="curve",
    col="model",
    kind="line",
    hue_order=order,
    style_order=order,
    palette=[sns.color_palette()[0], sns.color_palette()[1], "lightgrey"],
    dashes=["", "", (1, 1)],
)

for ax in g.axes[0]:
    # ax.vlines([0.01, 0.05, 0.1], 0, 1, linestyle="--", color="lightgray", linewidth=1.5,
              # label=r"Values in Fig. 1 (note that Fig. 1 reports $1 - \beta$ as `attack risk')")
    ax.set_ylabel(r"Attack FNR, $\beta$")
    ax.set_xlabel(r"Attack FPR, $\alpha$")

g.legend.set_title(None)
sns.move_legend(g, loc="lower right", bbox_to_anchor=(0.5, -0.25))

plt.savefig("../images/gpt2_trade_off_curves.pgf", bbox_inches="tight", format="pgf")
# -

# Utility plots.

# +
standard_delta = 1e-5
alphas = [0.01, 0.05, 0.1]

plot_data = []
for i, row in tqdm.tqdm(list(exp_metadata.iterrows())):
    standard_eps = get_epsilon(row.sigma, row.q, row.steps, standard_delta)

    for alpha in alphas:
        standard_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
            standard_eps, standard_delta, alpha=alpha
        )

        # FPR/FNR calibrated
        cal_beta = get_beta(row.sigma, row.q, row.steps, alpha=alpha)    
        print(f"{row.sigma=} {alpha=} {cal_beta=}")
        plot_data.append(
            dict(
                alpha=alpha,
                standard_beta=standard_beta,
                standard_tpr=1 - standard_beta,
                standard_eps=standard_eps,
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
    y="Attack risk",
    x="Accuracy",
    hue="Method",
    hue_order=["Standard calibration", "Attack risk calibration"],
    col=r"$\alpha$",
    kind="line",
    marker="o",
)

plt.savefig("../images/gpt2_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")
