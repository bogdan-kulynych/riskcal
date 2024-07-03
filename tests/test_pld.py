import numpy as np
import pytest
import itertools

import riskcal
from opacus import accountants
from scipy.stats import norm
from scipy.optimize import root_scalar


grid_step = 1e-3
sample_rate = 0.001
num_dpsgd_steps = 10000


@pytest.fixture
def accountant():
    yield riskcal.pld.CTDAccountant


@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps",
    [
        (0.01, 1, 1),
        (0.05, 1, 1),
        (0.10, 1, 1),
        (0.01, sample_rate, num_dpsgd_steps),
        (0.05, sample_rate, num_dpsgd_steps),
        (0.10, sample_rate, num_dpsgd_steps),
    ],
)
def test_advantage_calibration_correctness(advantage, sample_rate, num_steps):
    advantage_error = 0.01
    calibrated_mu = riskcal.pld.find_noise_multiplier_for_advantage(
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        advantage_error=advantage_error,
        grid_step=grid_step,
    )

    # Verify that mu is calibrated for (0, adv)-DP:
    assert riskcal.pld.get_advantage(
        noise_multiplier=calibrated_mu, sample_rate=sample_rate, num_steps=num_steps
    ) == pytest.approx(advantage, abs=advantage_error)

    # Solve advantage = 1 - 2 * alpha
    alpha = 0.5 * (1 - advantage)
    assert riskcal.pld.get_beta(
        alpha=alpha,
        noise_multiplier=calibrated_mu,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
    ) == pytest.approx(alpha, abs=advantage_error)


@pytest.mark.parametrize(
    "alpha, beta, sample_rate, num_steps",
    [
        # DP-SGD
        (0.1, 0.25, sample_rate, num_dpsgd_steps),
        (0.1, 0.50, sample_rate, num_dpsgd_steps),
        (0.1, 0.75, sample_rate, num_dpsgd_steps),
    ],
)
def test_err_rates_calibration_correctness(alpha, beta, sample_rate, num_steps):
    beta_error = 0.01

    calibrated_mu = riskcal.pld.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
        grid_step=grid_step,
    )

    expected_beta = riskcal.pld.get_beta(
        alpha=alpha,
        noise_multiplier=calibrated_mu,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
    )
    assert expected_beta == pytest.approx(beta, abs=beta_error)


@pytest.mark.skip("TODO: Investigate the discrepancy between direct and blackbox.")
@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps, method",
    [
        # DP-SGD
        (0.01, sample_rate, num_dpsgd_steps, "brent"),
        (0.05, sample_rate, num_dpsgd_steps, "brent"),
        (0.10, sample_rate, num_dpsgd_steps, "brent"),
    ],
)
def test_advantage_calibration_blackbox_vs_direct(
    accountant, advantage, sample_rate, num_steps, method
):
    eps_error = 1e-4
    advantage_error = 0.01

    direct_calibrated_mu = riskcal.pld.find_noise_multiplier_for_advantage(
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        advantage_error=advantage_error,
        grid_step=grid_step,
    )

    blackbox_calibrated_mu = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant=accountant,
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
    )

    assert direct_calibrated_mu == pytest.approx(
        blackbox_calibrated_mu, abs=advantage_error
    )


@pytest.mark.skip("Investigate big difference between direct and blackbox.")
@pytest.mark.parametrize(
    "alpha, beta, sample_rate, num_steps, method",
    [
        # DP-SGD
        (0.1, 0.25, sample_rate, num_dpsgd_steps, "brent"),
        (0.1, 0.50, sample_rate, num_dpsgd_steps, "brent"),
        (0.1, 0.75, sample_rate, num_dpsgd_steps, "brent"),
    ],
)
def test_err_rates_calibration_blackbox_vs_direct(
    accountant, alpha, beta, sample_rate, num_steps, method
):
    classical_delta = 1e-5
    eps_error = 1e-4
    beta_error = 0.01

    direct_calibrated_mu = riskcal.pld.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
        grid_step=grid_step,
    )

    blackbox_calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
    )
    blackbox_calibrated_mu = blackbox_calibration_result.noise_multiplier

    assert direct_calibrated_mu == pytest.approx(blackbox_calibrated_mu, abs=beta_error)
