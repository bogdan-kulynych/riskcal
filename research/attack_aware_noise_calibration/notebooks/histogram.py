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

# +
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from tqdm import auto as tqdm
from dp_accounting.pld import privacy_loss_distribution

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

adult_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"],
    skipinitialspace=True
)

hist = adult_data.education.value_counts()


# +
# Function to add Laplace noise
def add_laplace_noise(values, scale):
    noise = np.random.laplace(0, scale, len(values))
    return np.maximum((values + noise).round(), 0)

def add_gaussian_noise(values, scale):
    noise = np.random.normal(0, scale, len(values))
    return np.maximum((values + noise).round(), 0)

noisy_hist = add_laplace_noise(hist, 2)
noisy_hist = add_gaussian_noise(hist, 2)

# +
alphas = np.array([0.01, 0.05, 0.1])
# betas = [0.5, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
noise_multipliers = np.linspace(0.2, 5, 100)
num_measurements = 100
num_queries = 1
delta = 1 / (1.1 * len(adult_data))
methods = ["standard", "riskcal"]
mechanisms = ["gaussian"]
tol = 1e-5
data_chunks = []

def get_beta_using_method(noise_multiplier, alpha, method, mechanism):
    if mechanism == "laplace":
        pld = privacy_loss_distribution.from_laplace_mechanism(
            parameter=noise_multiplier,
            value_discretization_interval=1e-4,
        )
    elif mechanism == "gaussian":
        pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            value_discretization_interval=1e-4,
        )
        
    if method == "riskcal":
        return np.minimum(
            riskcal.pld.get_beta_from_pld(pld, alpha),
            riskcal.pld.get_alpha_from_pld(pld, alpha)
        )
    else:
        epsilon = pld.get_epsilon_for_delta(delta)
        return riskcal.utils.get_err_rate_for_epsilon_delta(epsilon=epsilon, delta=delta, alpha=alpha)

for alpha, noise_multiplier in tqdm.tqdm(list(itertools.product(alphas, noise_multipliers))):
    for mechanism, method in itertools.product(mechanisms, methods):
        beta = get_beta_using_method(noise_multiplier, alpha, method, mechanism)
        print(f"{alpha=:2f} {beta=:2f} {noise_multiplier=:2f} {method=}")
        
        for rep in range(num_measurements):
            errs = []
            for q in range(num_queries):
                if mechanism == "laplace":
                    noisy_hist = add_laplace_noise(hist, noise_multiplier)
                elif mechanism == "gaussian":
                    noisy_hist = add_gaussian_noise(hist, noise_multiplier)
                errs.append((noisy_hist - hist).abs().mean())
            err = np.mean(errs)
            
            data_chunks.append(dict(
                alpha = alpha,
                beta = beta,
                tpr = 1 - beta,
                err = err,
                noise_multiplier = noise_multiplier,
                mechanism = mechanism,
                method = method,
            ))

# +
sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .replace({
            "standard": "Standard calibration",
            "riskcal": "Attack risk calibration",
            "laplace": "Laplace",
            "gaussian": "Gaussian",
        })
        .rename(
            columns={
                "alpha": r"$\alpha$",
                "tpr": r"Attack risk",
                "err": "$L_1$ error",
                "noise_multiplier": "Noise scale",
                "method": "Method",
                "mechanism": "Mechanism",
            }
        )
    ),
    col=r"$\alpha$",
    x="Attack risk",
    y="$L_1$ error",
    hue="Method",
    # row="Mechanism",
    kind="line",
    hue_order=["Standard calibration", "Attack risk calibration"],
)

# plt.yscale("log")
plt.savefig("../images/histogram_err_rates_calibration.pgf", bbox_inches="tight", format="pgf")
