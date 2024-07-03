## riskcal

[![CI](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2407.02191-b31b1b.svg)](https://arxiv.org/abs/2407.02191)

---

⚠️  This is a research prototype. Avoid or be extra careful when using in production.

---

The library provides tools for calibrating the noise scale in (epsilon, delta)-DP mechanisms to one
of the two notions of operational attack risk (attack accuracy/advantage, or attack TPR and FPR) instead of the
(epsilon, delta) parameters, as well as for efficient measurement of these notions.
The library enables to reduce the noise scale at the same level of targeted attack risk.


### Using the Library

Install with:
```
pip install riskcal
```

#### Quickstart

##### Measuring the Exact f-DP / Trade-Off Curve for any DP Mechanism
To measure the attack trade-off curve (equivalent to attack's receiver-operating curve) for DP-SGD, you can run
```
import riskcal
import numpy as np

alphas = np.array([0.01, 0.05, 0.1])
betas = riskcal.pld.get_beta(
    alpha=alphas,
    noise_multiplier=noise_multiplier,
    sample_rate=sample_rate,
    num_steps=num_steps,
)
```

You can also get the trade-off curve for any DP mechanism [supported](https://github.com/google/differential-privacy/tree/0b109e959470c43e9f177d5411603b70a56cdc7a/python/dp_accounting)
by Google's DP accounting library, given its privacy loss distribution (PLD):
```
import riskcal
import numpy as np

alphas = np.array([0.01, 0.05, 0.1])
betas = riskcal.pld.get_beta_from_pld(pld, alpha=alphas)
```

##### Calibrating DP-SGD to attack FNR/FPR
To calibrate noise scale in DP-SGD to a given attack FPR (beta) and FNR (alpha), run:
```
import riskcal

noise_multiplier = riskcal.pld.find_noise_multiplier_for_err_rates(
    beta=0.2,
    alpha=0.01,
    sample_rate=sample_rate,
    num_steps=num_steps
)
```
