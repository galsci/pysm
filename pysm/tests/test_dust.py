import pytest
import numpy as np
import pysm
import pysm.component_models.galactic.dust as dust
import astropy.units as units
from astropy.units import UnitsError


def test_blackbody_ratio():
    nu_from = 100. * units.GHz
    nu_to = 400. * units.GHz
    temp = 20. * units.K

    nu_from_unitless = 100.
    nu_to_unitless = 400.
    temp_unitless = 20.

    nu_from_wrong_unit = 100. * units.K
    nu_to_wrong_unit = 400. * units.K
    temp_wrong_unit = 20. * units.s

    dust.blackbody_ratio(nu_to, nu_from, temp)
    dust.blackbody_ratio(nu_to, nu_from, temp)
    with pytest.raises(UnitsError):
        dust.blackbody_ratio(nu_to, nu_from_wrong_unit, temp)
    with pytest.raises(TypeError):
        dust.blackbody_ratio(nu_to, nu_from_unitless, temp)
    with pytest.raises(units.UnitsError):
        dust.blackbody_ratio(nu_to_wrong_unit, nu_from, temp)
    with pytest.raises(TypeError):
        dust.blackbody_ratio(nu_to_unitless, nu_from, temp)
    with pytest.raises(units.UnitsError):
        dust.blackbody_ratio(nu_to, nu_from, temp_wrong_unit)
    with pytest.raises(TypeError):
        dust.blackbody_ratio(nu_to, nu_from, temp_unitless)


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model_tag", ["d1"])
# @pytest.mark.parametrize("model_tag", ["d1", "d2", "d3"]) # FIXME activate testing for other models
def test_dust_model(model_tag, freq):

    model = pysm.preset_models(model_tag, nside=64)

    model_number = {"d1": 1, "d2": 6, "d3": 9}[model_tag]
    expected_output = pysm.read_map(
        "pysm_2_test_data/check{}therm_{}p0_64.fits".format(model_number, freq),
        64,
        field=(0, 1, 2),
    )

    frac_error = (expected_output - model.get_emission(freq * units.GHz)) / expected_output

    np.testing.assert_array_almost_equal(
        frac_error, np.zeros_like(frac_error), decimal=6
    )
