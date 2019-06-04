import pytest
import numpy as np
import pysm
import pysm.models.dust as dust
import astropy.units as units
from astropy.tests.helper import assert_quantity_allclose


def test_blackbody_ratio():
    nu_from = 100.0
    nu_to = 400.0
    temp = np.array([20.0])

    np.testing.assert_allclose(
        dust.blackbody_ratio(nu_to, nu_from, temp), np.array([10.77195547])
    )


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model_tag", ["d1"])
# @pytest.mark.parametrize("model_tag", ["d1", "d2", "d3"]) # FIXME activate testing for other models
def test_dust_model(model_tag, freq):

    model = pysm.Sky(preset_strings=[model_tag], nside=64)

    model_number = {"d1": 1, "d2": 6, "d3": 9}[model_tag]
    expected_output = pysm.read_map(
        "pysm_2_test_data/check{}therm_{}p0_64.fits".format(model_number, freq),
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    assert_quantity_allclose(
        expected_output, model.get_emission(freq * units.GHz), rtol=1e-5
    )
