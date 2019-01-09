import pytest
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
