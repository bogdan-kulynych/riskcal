import numpy as np
import pytest
import itertools

import riskcal
from opacus import accountants
from scipy.stats import norm
from scipy.optimize import root_scalar


sample_rate = 0.01
num_dpsgd_steps = 1000


@pytest.fixture
def accountant():
    yield accountants.prv.PRVAccountant


@pytest.mark.parametrize(
    "beta, sample_rate, num_steps",
    [
        # DP-SGD
        (0.25, sample_rate, num_dpsgd_steps),
        (0.50, sample_rate, num_dpsgd_steps),
        (0.75, sample_rate, num_dpsgd_steps),
    ],
)
def test_err_rates_calibration_correctness(beta, sample_rate, num_steps):
    alpha = 0.01
    beta_error = 0.001

    calibrated_mu = riskcal.pld.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
    )

    expected_beta = riskcal.pld.get_beta(
        alpha=alpha,
        noise_multiplier=calibrated_mu,
        sample_rate=sample_rate,
        num_steps=num_steps,
    )
    assert expected_beta == pytest.approx(beta, abs=beta_error)


# @pytest.mark.skip("Need to figure out deltas order issues from Connect the Dots.")
@pytest.mark.parametrize(
    "beta, sample_rate, num_steps, method",
    [
        # DP-SGD
        (0.25, sample_rate, num_dpsgd_steps, "brent"),
        (0.50, sample_rate, num_dpsgd_steps, "brent"),
        (0.75, sample_rate, num_dpsgd_steps, "brent"),
    ],
)
def test_err_rates_calibration_blackbox_vs_direct(
    accountant, beta, sample_rate, num_steps, method
):
    alpha = 0.01
    classical_delta = 1e-5
    delta_error = 0.001
    beta_error = 0.01

    direct_calibrated_mu = riskcal.pld.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
    )

    blackbox_calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
    )
    blackbox_calibrated_mu = blackbox_calibration_result.noise_multiplier

    assert direct_calibrated_mu == pytest.approx(blackbox_calibrated_mu, abs=beta_error)
