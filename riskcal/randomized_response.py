import numpy as np
from scipy.optimize import root_scalar


def get_adv_for_epsilon_delta(epsilon: float, delta: float) -> float:
    """Derive advantage from a given epsilon and delta.

    >>> np.round(get_adv_for_epsilon_delta(0., 0.001), 3)
    0.001
    """
    return (np.exp(epsilon) + 2 * delta - 1) / (np.exp(epsilon) + 1)


def get_epsilon_for_advantage(delta: float, adv: float) -> float:
    """Derive epsilon from a given advantage and delta.

    >>> np.round(get_epsilon_for_advantage(0.001, 0.5), 3)
    1.097
    """
    # We define the function to find the root of (f(eps) - 0) = 0
    root_result = root_scalar(
        lambda epsilon: get_adv_for_epsilon_delta(epsilon, delta) - adv,
        method="brentq",
        bracket=(0, 100),
    )

    if root_result.converged:
        return root_result.root
    else:
        raise ValueError("Root finding did not converge.")


def get_epsilon_for_err_rates(delta: float, alpha: float, beta: float):
    """Derive epsilon from given FPR/FNR error rates and delta.

    >>> np.round(get_epsilon_for_err_rates(0.001, 0.001, 0.8), 3)
    5.293
    """
    epsilon1 = np.log((1 - delta - alpha) / beta)
    epsilon2 = np.log((1 - delta - beta) / alpha)
    return np.maximum.reduce([epsilon1, epsilon2, np.zeros_like(epsilon1)])


def get_delta_for_err_rates(epsilon: float, alpha: float, beta: float):
    """Derive delta from given FPR/FNR error rates and epsilon.

    >>> np.round(get_delta_for_err_rates(1.0, 0.001, 0.8), 3)
    0.197
    """
    delta1 = (1 - beta) - np.exp(epsilon) * alpha
    delta2 = (1 - alpha) - np.exp(epsilon) * beta
    return np.maximum.reduce([delta1, delta2, np.zeros_like(delta1)])
