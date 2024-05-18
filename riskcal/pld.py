import numpy as np
import matplotlib.pyplot as plt
from dp_accounting.pld import privacy_loss_distribution
from scipy.optimize import root_scalar


def _get_domain_and_pmf_from_pld(pld):
    pld = pld.to_dense_pmf()
    pmf = pld._probs
    domain = (pld._lower_loss + np.arange(len(pld._probs))) * pld._discretization
    return domain, pmf


def _get_lower_loss_and_pmf_from_pld(pld):
    pld = pld.to_dense_pmf()
    pmf = pld._probs
    lower_loss = pld._lower_loss
    return lower_loss, pmf


def _get_dpsgd_composed_plrv_pmfs(
    noise_multiplier, sample_rate, num_steps, grid_step=0.002
):
    google_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    )
    composed_google_pld = google_pld.self_compose(num_steps)

    lower_loss_Y, pmf_Y = _get_lower_loss_and_pmf_from_pld(
        composed_google_pld._pmf_remove
    )
    lower_loss_Z, pmf_Z = _get_lower_loss_and_pmf_from_pld(composed_google_pld._pmf_add)

    return lower_loss_Z, pmf_Z, lower_loss_Y, pmf_Y


def get_beta(
    alpha: float,
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=0.002,
):
    """
    Find FNR for a given FPR in DP-SGD.

    Arguments:
        alpha: Target FPR
        noise_multiplier: DP-SGD noise multiplier
        sample_rate: Subsampled Gaussian sampling rate
        num_steps: Number of steps
        gird_step: Step size of the discretization grid
    """
    num_steps = int(num_steps)

    # Get the composed PLRVs. Using paper notation, note that Z = -X
    lower_loss_Z, pmf_Z, lower_loss_Y, pmf_Y = _get_dpsgd_composed_plrv_pmfs(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
    )
    # Get the discrete points of alpha, beta
    alphas = np.cumsum(pmf_Z) - pmf_Z
    betas = np.cumsum(pmf_Y)

    # Binary search to find the right index.
    idx_Z = np.searchsorted(alphas, alpha) - 1

    # Sanity check: did we find the correct index?
    # Note that the alphas are in descending order.
    assert np.all(alphas[idx_Z] < alpha) and np.all(alpha < alphas[idx_Z + 1])

    # Find gamma.
    gamma = (alpha - alphas[idx_Z]) / pmf_Z[idx_Z]

    # Sanity check: gamma should be a positive and less than 1
    assert np.all(0 < gamma) and np.all(gamma < 1)

    # Get index in the Y world
    idx_Y = -lower_loss_Z - lower_loss_Y - idx_Z

    # Compute beta.
    beta = betas[idx_Y] - gamma * pmf_Y[idx_Y]

    # Sanity check: did we somehow go over to the next beta?
    assert np.all(betas[idx_Y - 1] < beta) and np.all(beta < betas[idx_Y])

    return beta


def find_noise_multiplier_for_err_rates(
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    grid_step: float = 0.002,
    mu_max=100.0,
    beta_error=0.001,
):
    """
    Find a noise multiplier that satisfies a given target epsilon.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param alpha: Attack FPR bound
    :param beta: Attack FNR bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param grid_step: Discretization grid step
    :param delta: Value of DP delta
    :param float mu_max: Maximum value of noise multiplier of the search.
    """

    def _get_beta(mu):
        return get_beta(
            noise_multiplier=mu,
            alpha=alpha,
            sample_rate=sample_rate,
            num_steps=num_steps,
            grid_step=grid_step,
        )

    mu_R = 1.0
    beta_R = 0.0
    while beta_R < beta:
        mu_R *= np.sqrt(2)
        try:
            beta_R = _get_beta(mu_R)
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError(
                "Finding a suitable noise multiplier has not converged. "
                "Try decreasing target beta or decreasing sample rate."
            )

    mu_L = mu_R
    beta_L = beta_R
    while beta_L > beta:
        mu_L /= np.sqrt(2)
        beta_L = _get_beta(mu_L)

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1] - bracket[0]) * 0.01

        mu_guess = root_scalar(
            lambda mu: _get_beta(mu) - beta,
            bracket=bracket,
            xtol=mu_err,
        ).root
        bracket = [mu_guess - mu_err, mu_guess + mu_err]
        beta_low = _get_beta(mu_guess - mu_err)
        beta_up = _get_beta(mu_guess + mu_err)
        has_converged = (beta_up - beta_low) < beta_error

    assert _get_beta(bracket[1]) > beta - beta_error
    return bracket[1]
