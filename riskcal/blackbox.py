from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import root_scalar, minimize_scalar

from . import utils


def find_noise_multiplier_for_epsilon_delta(
    accountant: "opacus.accountants.accountant.IAccountant",
    sample_rate: float,
    num_steps: int,
    epsilon: float,
    delta: float,
    eps_error: float = 0.001,
    mu_max: float = 100.0,
    **accountant_kwargs,
) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param accountant: Opacus-compatible accountant
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param epsilon: Desired target epsilon
    :param delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    :param float mu_max: Maximum value of noise multiplier of the search.
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """

    def compute_epsilon(mu: float) -> float:
        acc = accountant()
        for step in range(num_steps):
            acc.step(noise_multiplier=mu, sample_rate=sample_rate)

        return acc.get_epsilon(delta=delta, **accountant_kwargs)

    mu_R = 1.0
    eps_R = float("inf")
    while eps_R > epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError(
                "Finding a suitable noise multiplier has not converged. "
                "Try increasing target epsilon or decreasing sample rate."
            )

    mu_L = mu_R
    eps_L = eps_R
    while eps_L <= epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1] - bracket[0]) * 0.01
        assert mu_err > 0
        mu_guess = root_scalar(
            lambda mu: compute_epsilon(mu) - epsilon,
            bracket=bracket,
            xtol=mu_err,
        ).root
        bracket = [mu_guess - mu_err, mu_guess + mu_err]
        eps_up = compute_epsilon(mu_guess - mu_err)
        eps_low = compute_epsilon(mu_guess + mu_err)
        has_converged = (eps_up - eps_low) < eps_error
    assert compute_epsilon(bracket[1]) < epsilon + eps_error
    return bracket[1]


def find_noise_multiplier_for_advantage(
    accountant: "opacus.accountants.accountant.IAccountant",
    advantage: float,
    sample_rate: float,
    num_steps: float,
    eps_error=0.001,
    mu_max=100.0,
    **accountant_kwargs,
):
    """
    Find a noise multiplier that satisfies given levels of attack advantage.

    :param accountant: Opacus-compatible accountant
    :param advantage: Attack advantage bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param float eps_error: Error allowed for final epsilon
    :param float mu_max: Maximum value of noise multiplier of the search
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """
    return find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=0.0,
        delta=advantage,
        eps_error=eps_error,
        mu_max=mu_max,
        **accountant_kwargs,
    )


class _ErrRatesAccountant:
    def __init__(
        self,
        accountant,
        alpha,
        beta,
        sample_rate,
        num_steps,
        eps_error,
        mu_max=100.0,
        **accountant_kwargs,
    ):
        self.accountant = accountant
        self.alpha = alpha
        self.beta = beta
        self.sample_rate = sample_rate
        self.num_steps = num_steps
        self.eps_error = eps_error
        self.mu_max = mu_max
        self.accountant_kwargs = accountant_kwargs

    def find_noise_multiplier(self, delta):
        epsilon = utils.get_epsilon_for_err_rates(delta, self.alpha, self.beta)
        try:
            mu = find_noise_multiplier_for_epsilon_delta(
                epsilon=epsilon,
                delta=delta,
                accountant=self.accountant,
                sample_rate=self.sample_rate,
                num_steps=self.num_steps,
                eps_error=self.eps_error,
                mu_max=self.mu_max,
                **self.accountant_kwargs,
            )
            return mu

        except RuntimeError as e:
            warnings.warn(
                f"Error occured in grid search w/ {epsilon=:.4f} {delta=:.4f}"
            )
            warnings.warn(e)
            return np.inf


def _find_noise_profile(
    err_rates_acct_obj: _ErrRatesAccountant,
    max_delta: float,
    sample_rate: float,
    num_steps: float,
    delta_error=0.01,
    eps_error=0.001,
    mu_max=100.0,
    **accountant_kwargs,
):
    delta_vals = np.linspace(
        delta_error, max_delta, int((max_delta - delta_error) / delta_error)
    )
    if len(delta_vals) == 0:
        raise ValueError("Grid resolution too low. Try increasing delta_error.")

    noise_vals = np.array([np.inf] * len(delta_vals))
    for i, delta in enumerate(delta_vals):
        noise_vals[i] = err_rates_acct_obj.find_noise_multiplier(delta)

    return delta_vals, noise_vals


@dataclass
class CalibrationResult:
    """
    Result of generic calibration.
    """

    noise_multiplier: float
    calibration_epsilon: float
    calibration_delta: float


def find_noise_multiplier_for_err_rates(
    accountant: "opacus.accountants.accountant.IAccountant",
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    delta_error=0.01,
    eps_error=0.001,
    mu_max=100.0,
    method="brent",
    **accountant_kwargs,
):
    """
    Find a noise multiplier that limits attack FPR/FNR rates.

    :param accountant: Opacus-compatible accountant
    :param alpha: Attack FPR bound
    :param beta: Attack FNR bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param float delta_error: Error allowed for delta used for calibration
    :param float eps_error: Error allowed for final epsilon
    :param float mu_max: Maximum value of noise multiplier of the search
    :param str method: Optimization method. One of ['brent', 'grid_search']
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """
    if alpha + beta >= 1:
        raise ValueError(
            f"The guarantees are vacuous when alpha + beta >= 1. Got {alpha=}, {beta=}"
        )

    max_delta = 1 - alpha - beta
    err_rates_acct_obj = _ErrRatesAccountant(
        accountant=accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
        mu_max=mu_max,
        **accountant_kwargs,
    )

    if max_delta < delta_error:
        raise ValueError(f"{delta_error=} too low for the requested error rates.")

    if method == "brent":
        opt_result = minimize_scalar(
            err_rates_acct_obj.find_noise_multiplier,
            bounds=[delta_error, max_delta],
            options=dict(xatol=delta_error),
        )
        if not opt_result.success:
            raise RuntimeError(f"Optimization failed: {opt_result.message}")
        calibration_delta = opt_result.x
        noise_multiplier = opt_result.fun

    elif method == "grid_search":
        delta_vals, noise_vals = _find_noise_profile(
            err_rates_acct_obj=err_rates_acct_obj,
            max_delta=max_delta,
            sample_rate=sample_rate,
            num_steps=num_steps,
            delta_error=delta_error,
            eps_error=eps_error,
            mu_max=mu_max,
        )

        noise_multiplier = noise_vals.min()
        calibration_delta = delta_vals[np.argmin(noise_vals)]

    return CalibrationResult(
        noise_multiplier=noise_multiplier,
        calibration_delta=calibration_delta,
        calibration_epsilon=utils.get_epsilon_for_err_rates(
            calibration_delta, alpha, beta
        ),
    )
