## Attack-Aware Noise Calibration for Differential Privacy

[![CI](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml)

---

⚠️  This is a research prototype. Avoid or be extra careful when using in production.

---

The library provides tools for calibrating the noise scale in (epsilon, delta)-DP mechanisms to one
of the two notions of operational attack risk (attack accuracy/advantage, or attack TPR and FPR) instead of the
(epsilon, delta) parameters. This enables to reduce the noise scale at the same level of targeted attack risk.


### Using the Library

Install with:
```
pip install riskcal
```

#### Quickstart

First, set up an [opacus](https://github.com/pytorch/opacus/)-compatible privacy accountant:
```
import riskcal
from opacus import accountants
accountant = accountants.rdp.RDPAccountant
```

Calibrate to a given attack advantage (`target_adv`):

```
calibration_result = riskcal.blackbox.find_noise_multiplier_for_advantage(
    accountant=accountant,
    advantage=target_adv,
    sample_rate=sample_rate,
    num_steps=num_steps,
    delta_error=delta_error
)
```

Calibrate noise to a given attack TPR (`target_tpr`) at FPR = 0.01:

```
import riskcal

calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
    accountant=accountant,
    beta=1 - target_tpr,
    alpha=0.01,
    sample_rate=sample_rate,
    num_steps=num_steps,
    delta_error=delta_error
)
```

Retrieve the calibrated noise multiplier:
```
calibration_result.noise_multiplier
```

### Installing Dependencies for Reproducing Experiments in the Paper

```
poetry install --with experiments --with dev
```

### Reproducing Plots in the Paper

Run:
```
poetry run jupytext --to ipynb experiments/notebooks/*.py
poetry run jupyter notebook
```

Then, run each notebook in Jupyter.
