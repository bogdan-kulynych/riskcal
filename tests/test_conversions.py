import pytest
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

import riskcal


@pytest.fixture
def plrv_data():
    mu = 2.0
    pld = from_gaussian_mechanism(standard_deviation=1.0 / mu)
    return mu, pld, riskcal.conversions.plrvs_from_pld(pld)


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_beta_matches_analytic_curve(plrv_data, alpha):
    mu, pld, plrvs = plrv_data
    analytic_beta = riskcal.conversions.get_beta_for_mu(mu, alpha)
    assert pytest.approx(analytic_beta) == riskcal.plrv.get_beta(plrvs, alpha=alpha)
    assert pytest.approx(analytic_beta) == riskcal.conversions.get_beta_from_pld(
        pld, alpha=alpha
    )


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_advantage_matches_analytic_matching_expression(plrv_data, alpha):
    mu, pld, plrvs = plrv_data
    analytic_advantage = riskcal.conversions.get_advantage_for_mu(mu)
    assert pytest.approx(
        analytic_advantage
    ) == riskcal.conversions.get_advantage_from_pld(pld)
