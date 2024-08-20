from copy import deepcopy
from functools import reduce
from typing import Union
from dp_accounting.pld import privacy_loss_distribution
from scipy.optimize import root_scalar
import numpy as np

from riskcal import conversions


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

    def get_pld(self, grid_step=1e-4, use_connect_dots=True):
        plds = []
        for noise_multiplier, sample_rate, num_steps in self.history:
            pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=noise_multiplier,
                sampling_prob=sample_rate,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=grid_step,
            )
            plds.append(pld.self_compose(num_steps))

        return reduce(lambda a, b: a.compose(b), plds)

    def get_epsilon(self, *, delta, **kwargs):
        pld = self.get_pld(**kwargs)
        return pld.get_epsilon_for_delta(delta)

    def get_beta(self, *, alpha, **kwargs):
        pld = self.get_pld(**kwargs)
        return conversions.get_beta_from_pld(pld, alpha)

    def get_advantage(self, **kwargs):
        pld = self.get_pld(**kwargs)
        return conversions.get_advantage_from_pld(pld)

    def __len__(self):
        total = 0
        for _, _, steps in self.history:
            total += steps
        return total

    def mechanism(self):
        return "ctd"

    # The following methods are copied from https://opacus.ai/api/_modules/opacus/accountants/accountant.html#IAccountant
    # to avoid the direct dependence on the opacus package.
    def get_optimizer_hook_fn(self, sample_rate: float):
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
            destination = {}
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


def get_advantage_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=1e-4,
):
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)
    return conversions.get_advantage_from_pld(pld)


def get_beta_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    alpha: Union[float, np.ndarray],
    grid_step=1e-4,
):
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)
    return conversions.get_beta_from_pld(pld, alpha)


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
        return get_beta_for_dpsgd(
            noise_multiplier=mu,
            sample_rate=sample_rate,
            num_steps=num_steps,
            alpha=alpha,
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
