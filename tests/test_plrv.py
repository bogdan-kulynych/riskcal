import pytest
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

import riskcal


@pytest.fixture
def plrvs():
    pld = from_gaussian_mechanism(standard_deviation=1.0, sampling_prob=1.0)
    return riskcal.conversions.plrvs_from_pld(pld)


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_beta(plrvs, alpha):
    real_beta = riskcal.conversions.get_beta_from_gdp(1.0, alpha)
    assert pytest.approx(real_beta) == riskcal.plrv.get_beta(plrvs, alpha=alpha)
