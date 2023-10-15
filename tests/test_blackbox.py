import numpy as np
import pytest
import itertools

import riskcal
from opacus import accountants


@pytest.fixture
def accountant():
    yield accountants.rdp.RDPAccountant


sample_rate = 0.001
num_dpsgd_steps = 100


@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps",
    [
        (0.1, 1, 1),
        (0.1, 1, 1),
        (0.1, 1, 1),
        (0.25, sample_rate, num_dpsgd_steps),
        (0.25, sample_rate, num_dpsgd_steps),
        (0.25, sample_rate, num_dpsgd_steps),
    ],
)
def test_adv_calibration(accountant, advantage, sample_rate, num_steps):
    calibrated_mu = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant, advantage=advantage, sample_rate=sample_rate, num_steps=num_steps
    )

    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)

    # Verify that mu is calibrated for (0, adv)-DP:
    assert acct_obj.get_epsilon(delta=advantage) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize(
    "beta, sample_rate, num_steps, method",
    [
        # (0.25, 1, 1, "brent"),
        (0.25, sample_rate, num_dpsgd_steps, "brent"),
        (0.10, sample_rate, num_dpsgd_steps, "brent"),
    ],
)
def test_err_rates_adv_calibration_equivalence(
    accountant, beta, sample_rate, num_steps, method
):
    alpha = beta
    advantage = 1 - alpha - beta
    classical_delta = 1e-5
    delta_error = 0.001
    eps_error = 0.01

    assert alpha + beta < 1
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        mu_max=100.0,
        method=method,
    )

    noise_multiplier_for_adv = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant,
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        mu_max=100.0,
    )
    assert calibration_result.calibration_epsilon == pytest.approx(0, abs=eps_error)
    assert calibration_result.noise_multiplier == pytest.approx(
        noise_multiplier_for_adv, abs=eps_error
    )


@pytest.mark.parametrize(
    "beta, sample_rate, num_steps, method",
    [
        # Gaussian mechanism:
        (0.5, 1, 1, "brent"),
        (0.5, 1, 1, "grid_search"),
    ],
)
def test_err_rates_calibration_noise_profile(
    accountant, beta, sample_rate, num_steps, method
):
    alpha = 0.01
    classical_delta = 1e-5
    delta_error = 0.001
    eps_error = 0.01

    assert alpha + beta < 1
    max_delta = 1 - alpha - beta
    err_rates_acct_obj = riskcal.blackbox._ErrRatesAccountant(
        accountant=accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
        mu_max=100.0,
    )

    delta_vals, noise_vals = riskcal.blackbox._find_noise_profile(
        err_rates_acct_obj,
        max_delta=max_delta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
    )
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        mu_max=100.0,
        method=method,
    )
    assert calibration_result.noise_multiplier == pytest.approx(
        min(noise_vals), abs=eps_error
    )


@pytest.mark.parametrize(
    "beta, sample_rate, num_steps, method",
    [
        # Gaussian mechanism:
        (0.25, 1, 1, "brent"),
        (0.50, 1, 1, "brent"),
        (0.75, 1, 1, "brent"),
        (0.25, 1, 1, "grid_search"),
        (0.50, 1, 1, "grid_search"),
        (0.75, 1, 1, "grid_search"),
        # DP-SGD
        (0.25, sample_rate, num_dpsgd_steps, "brent"),
        (0.50, sample_rate, num_dpsgd_steps, "brent"),
        (0.75, sample_rate, num_dpsgd_steps, "brent"),
    ],
)
def test_err_rates_calibration_correctness(
    accountant, beta, sample_rate, num_steps, method
):
    alpha = 0.01
    classical_delta = 1e-5
    delta_error = 0.001
    eps_error = 0.01

    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
    )
    calibrated_mu = calibration_result.noise_multiplier
    calibrated_delta = calibration_result.calibration_delta

    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)

    epsilon = acct_obj.get_epsilon(delta=calibrated_delta)
    expected_epsilon = calibration_result.calibration_epsilon

    print(f"CHECK 1: {alpha=}, {beta=} // {epsilon=}, {expected_epsilon=}")
    assert epsilon == pytest.approx(expected_epsilon, abs=eps_error)

    obtained_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
        epsilon, calibrated_delta, alpha
    )
    print(f"CHECK 2: {alpha=}, {beta=} // {epsilon=}, {obtained_beta=}")
    assert beta == pytest.approx(obtained_beta, abs=delta_error)


@pytest.mark.parametrize(
    "epsilon, sample_rate, num_steps",
    [
        (1.0, 1.0, 1),
        (1.0, 1.0, 1),
        (1.0, 1.0, 1),
        (4.0, sample_rate, num_dpsgd_steps),
        (4.0, sample_rate, num_dpsgd_steps),
        (4.0, sample_rate, num_dpsgd_steps),
    ],
)
def test_err_rates_calibration_improvement(accountant, epsilon, sample_rate, num_steps):
    alpha = 0.1
    delta = 1e-5
    delta_error = 0.05
    eps_error = 0.01
    method = "brent"

    standard_mu = riskcal.blackbox.find_noise_multiplier_for_epsilon_delta(
        accountant,
        epsilon=epsilon,
        delta=delta,
        sample_rate=sample_rate,
        num_steps=num_steps,
    )

    # What is the FNR at alpha = 0.1 for the target epsilon?
    beta = riskcal.utils.get_err_rate_for_epsilon_delta(epsilon, delta, alpha)
    # Let's calibrate directly for these alpha / beta:
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        method=method,
    )
    calibrated_mu = calibration_result.noise_multiplier
    calibrated_delta = calibration_result.calibration_delta

    # We should get better less noise with direct calibration:
    assert standard_mu / calibrated_mu > 1.5

    # Check that alpha beta guarantees are correct.
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)
    obtained_epsilon = acct_obj.get_epsilon(delta=calibrated_delta)

    obtained_beta = riskcal.utils.get_err_rate_for_epsilon_delta(
        obtained_epsilon, calibrated_delta, alpha
    )
    assert beta == pytest.approx(obtained_beta, abs=delta_error)
