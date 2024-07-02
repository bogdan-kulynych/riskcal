from functools import reduce
from typing import Union

import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from scipy.optimize import root_scalar


class CTDAccountant:
    """
    Opacus-compatible Connect the Dots accountant.
    """
    def __init__(self):
        self.history = []

    def step(self, *, noise_multiplier, sample_rate):
        if len(self.history) > 1:
            prev_noise_multiplier, prev_sample_rate, prev_steps = self.history[-1]
            if (
                prev_noise_multiplier == noise_multiplier
                and prev_sample_rate == sample_rate
            ):
                self.history[-1] = (noise_multiplier, sample_rate, prev_steps + 1)
                return
        self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(self, *, delta, grid_step=1e-4, **kwargs):
        plds = []
        for noise_multiplier, sample_rate, num_steps in self.history:
            pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=noise_multiplier,
                sampling_prob=sample_rate,
                use_connect_dots=True,
                value_discretization_interval=grid_step,
            )
            plds.append(pld.self_compose(num_steps))

        composed_pld = reduce(lambda a, b: a.compose(b), plds)
        return composed_pld.get_epsilon_for_delta(delta)

    def __len__(self):
        total = 0
        for _, _, steps in self.history:
            total += steps
        return total

    def mechanism(self):
        return "ctd"

    # The following methods are copied from https://opacus.ai/api/_modules/opacus/accountants/accountant.html#IAccountant
    # to avoid the direct dependence on the opacus package.
    def get_optimizer_hook_fn(
        self, sample_rate: float
    ):
        """
        Returns a callback function which can be used to attach to DPOptimizer
        Args:
            sample_rate: Expected sampling rate used for accounting
        """

        def hook_fn(optim):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        return hook_fn

    def state_dict(self, destination=None):
        """
        Returns a dictionary containing the state of the accountant.
        Args:
            destination: a mappable object to populate the current state_dict into.
                If this arg is None, an OrderedDict is created and populated.
                Default: None
        """
        if destination is None:
            destination = OrderedDict()
        destination["history"] = deepcopy(self.history)
        destination["mechanism"] = self.__class__.mechanism
        return destination

    def load_state_dict(self, state_dict):
        """
        Validates the supplied state_dict and populates the current
        Privacy Accountant's state dict.

        Args:
            state_dict: state_dict to load.

        Raises:
            ValueError if supplied state_dict is invalid and cannot be loaded.
        """
        if state_dict is None or len(state_dict) == 0:
            raise ValueError(
                "state dict is either None or empty and hence cannot be loaded"
                " into Privacy Accountant."
            )
        if "history" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `history`."
                " Cannot be loaded into Privacy Accountant."
            )
        if "mechanism" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `mechanism`."
                " Cannot be loaded into Privacy Accountant."
            )
        if self.__class__.mechanism != state_dict["mechanism"]:
            raise ValueError(
                f"state_dict of {state_dict['mechanism']} cannot be loaded into "
                f" Privacy Accountant with mechanism {self.__class__.mechanism}"
            )
        self.history = state_dict["history"]


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


def get_beta_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    alpha: Union[float, np.ndarray],
):
    lower_loss_Y, pmf_Y = _get_lower_loss_and_pmf_from_pld(
        pld._pmf_remove
    )
    lower_loss_Z, pmf_Z = _get_lower_loss_and_pmf_from_pld(pld._pmf_add)

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


def get_beta(
    alpha: Union[float, np.ndarray],
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=1e-4,
):
    """
    Find FNR for a given FPR in DP-SGD.

    Arguments:
        alpha: Target FPR, either a single float or a numpy array
        noise_multiplier: DP-SGD noise multiplier
        sample_rate: Subsampled Gaussian sampling rate
        num_steps: Number of steps
        gird_step: Step size of the discretization grid
    """
    num_steps = int(num_steps)

    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    )
    pld = pld.self_compose(num_steps)
    return get_beta_from_pld(pld, alpha=alpha)


def get_advantage(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=1e-4,
):
    google_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    )
    composed_google_pld = google_pld.self_compose(num_steps)
    return composed_google_pld.get_delta_for_epsilon(0)


def find_noise_multiplier_for_err_rates(
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    grid_step: float = 1e-4,
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


def find_noise_multiplier_for_advantage(
    advantage: float,
    sample_rate: float,
    num_steps: float,
    grid_step: float = 1e-4,
    mu_max=100.0,
    advantage_error=0.001,
):
    """
    Find a noise multiplier that satisfies a given target advantage.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param alpha: Attack FPR bound
    :param beta: Attack FNR bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param grid_step: Discretization grid step
    :param delta: Value of DP delta
    :param float mu_max: Maximum value of noise multiplier of the search.
    """
    # Solve advantage = 1 - 2 * fp
    fp = -0.5 * (advantage - 1)
    return find_noise_multiplier_for_err_rates(
        alpha=fp,
        beta=fp,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
        mu_max=mu_max,
        beta_error=advantage_error,
    )
