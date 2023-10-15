import warnings

import numpy as np
from scipy.optimize import root_scalar

from . import utils


def find_noise_multiplier_for_epsilon_delta(
    accountant,
    sample_rate: float,
    num_steps: int,
    epsilon: float,
    delta: float,
    eps_error: float = 0.01,
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

        return acc.get_epsilon(delta=delta)

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
                "Try increasing target epsilon or decreasing sampling probability."
            )

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1] - bracket[0]) * 0.01
        mu_guess = root_scalar(
            lambda mu: compute_epsilon(mu) - epsilon,
            bracket=bracket,
            xtol=mu_err,
        ).root
        bracket = [mu_guess - mu_err, mu_guess + mu_err]
        eps_up = compute_epsilon(mu_guess - mu_err)
        eps_low = compute_epsilon(mu_guess + mu_err)
        has_converged = (eps_up - eps_low) < 2 * eps_error
    assert compute_epsilon(bracket[1]) < epsilon + eps_error
    return bracket[1]


def find_noise_multiplier_for_advantage(
    accountant,
    advantage: float,
    sample_rate: float,
    num_steps: float,
    eps_error=0.01,
    nu_max=100.0,
    **accountant_kwargs,
):
    """
    Find a noise multiplier that satisfies given levels of adversary's advantage.

    :param accountant: Opacus-compatible accountant
    :param advantage: Adversary's advantage
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
        mu_max=nu_max,
        **accountant_kwargs,
    )


def _find_noise_profile(
    accountant,
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    delta_error=0.05,
    eps_error=0.01,
    nu_max=100.0,
    **accountant_kwargs,
):
    if alpha + beta >= 1:
        raise ValueError(
            f"The guarantees are vacuous when alpha + beta >= 1. Got {alpha=}, {beta=}"
        )
    max_delta = 1 - alpha - beta

    delta_vals = np.linspace(
        delta_error, max_delta, int((max_delta - delta_error) / delta_error)
    )
    noise_vals = np.array([np.inf] * len(delta_vals))
    for i, delta in enumerate(delta_vals):
        epsilon = utils.get_epsilon_for_err_rates(delta, alpha, beta)
        try:
            noise_candidate = find_noise_multiplier_for_epsilon_delta(
                accountant=accountant,
                sample_rate=sample_rate,
                num_steps=num_steps,
                epsilon=epsilon,
                delta=delta,
                eps_error=eps_error,
                mu_max=nu_max,
            )
            noise_vals[i] = noise_candidate
            # print(f"{epsilon=:.3f} {delta=:.3f} {noise_candidate=:.3f}")
        except RuntimeError as e:
            warnings.warn(
                f"Error occured on grid search w/ {epsilon=:.4f} {delta=:.4f}"
            )
            warnings.warn(e)

    return delta_vals, noise_vals


def find_noise_multiplier_for_err_rates(
    accountant,
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    delta_error=0.01,
    eps_error=0.001,
    nu_max=100.0,
    **accountant_kwargs,
):
    """
    Find a noise multiplier that satisfies given levels of adversary's FPR/FNR rates.

    :param accountant: Opacus-compatible accountant
    :param alpha: Adversary's FPR
    :param beta: Adversary's FNR
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param float delta_error: Error for delta grid discretization
    :param float eps_error: Error allowed for final epsilon
    :param float mu_max: Maximum value of noise multiplier of the search
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """
    _, noise_vals = _find_noise_profile(
        accountant=accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        eps_error=eps_error,
        nu_max=nu_max,
    )
    best_noise = min(noise_vals)
    return best_noise
