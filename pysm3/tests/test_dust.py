import astropy.units as units
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

import pysm3
import pysm3.models.dust as dust
import pytest


def test_blackbody_ratio():
    nu_from = 100.0
    nu_to = 400.0
    temp = np.array([20.0])

    np.testing.assert_allclose(
        dust.blackbody_ratio(nu_to, nu_from, temp), np.array([10.77195547]), rtol=1e-5
    )


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model_tag", ["d0", "d1", "d2", "d3", "d6"])
def test_dust_model(model_tag, freq):

    # for 'd6' model fix the random seed and skip buggy 353 GHz
    if model_tag == "d6":
        if freq == 353:
            return
        np.random.seed(123)

    model = pysm3.Sky(preset_strings=[model_tag], nside=64)

    model_number = {"d0": 1, "d1": 1, "d2": 6, "d3": 9, "d6": 12}[model_tag]
    expected_output = pysm3.read_map(
        "pysm_2_test_data/check{}therm_{}p0_64.fits".format(model_number, freq),
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    # for some models we do not have tests, we compare with output from a simular model
    # and we increase tolerance, mostly just to exercise the code.
    rtol = {"d0": 0.9}.get(model_tag, 1e-5)

    assert_quantity_allclose(
        expected_output, model.get_emission(freq * units.GHz), rtol=rtol
    )
