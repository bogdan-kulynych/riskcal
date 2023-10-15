import numpy as np
from scipy.optimize import root_scalar


def get_adv_for_epsilon_delta(epsilon: float, delta: float) -> float:
    """Advantage from eps, delta."""
    return (np.exp(epsilon) + 2 * delta - 1) / (np.exp(epsilon) + 1)


def get_epsilon_for_advantage(delta: float, adv: float) -> float:
    """Derive epsilon for a given advantage and delta."""
    # We define the function to find the root of (f(eps) - 0) = 0
    root_result = root_scalar(
        lambda epsilon: get_adv_for_epsilon_delta(epsilon, delta) - adv, method="brentq", bracket=(0, 100)
    )

    if root_result.converged:
        return root_result.root
    else:
        raise ValueError("Root finding did not converge.")


def get_epsilon_for_err_rates(delta: float, alpha: float, beta: float):
    """Derive epsilon for given FPR/FNR error rates."""
    epsilon1 = np.log((1 - delta - alpha) / beta)
    epsilon2 = np.log((1 - delta - beta) / alpha)
    return max(epsilon1, epsilon2, 0.0)


def get_delta_for_err_rates(epsilon: float, alpha: float, beta: float):
    """Derive delta for given FPR/FNR error rates."""
    delta1 = (1 - beta) - np.exp(epsilon) * alpha
    delta2 = (1 - alpha) - np.exp(epsilon) * beta
    return max(delta1, delta2, 0)


def get_err_rate_for_epsilon_delta(epsilon, delta, alpha):
    """Get error rate for a given epsilon, delta, and the other type error rate."""
    # See, e.g., Eq. 5 in https://arxiv.org/abs/1905.02383
    return max(
        0,
        1 - delta - np.exp(epsilon) * alpha,
        np.exp(-epsilon) * (1 - delta - alpha),
    )

