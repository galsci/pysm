import healpy as hp
import numpy as np
import pytest

from pysm3 import units
from pysm3.models.template import read_map


@pytest.fixture
def setUp():
    T_CMB = 100.0 * units.K_CMB
    T_RJ = 100.0 * units.K_RJ
    freqs = 100.0 * units.GHz
    nside = 256
    npix = hp.nside2npix(nside)
    test_map_RJ = np.random.randn(npix) * units.K_RJ
    test_map_CMB = np.random.randn(npix) * units.K_CMB
    test_map_dimless = units.Quantity(np.random.randn(npix), "")
    return T_CMB, T_RJ, freqs, test_map_RJ, test_map_CMB, test_map_dimless


def test_conversion(setUp):
    """ Here we test that the numerical value of the conversion is correct.
    The mathematical form is

    ..math::
    I_\\nu = \\frac{2 \\nu^2 k T_{\\rm RJ}}{c^2} \\\\
    I_\\nu = T_{\\rm CMB} B^\\prime_\\nu(T_{\\rm CMB, 0})

    so, eliminating the flux in this equation:

    ..math::
    T_{\\rm RJ} / T_{\\rm CMB} = \\frac{c^2}{2 \\nu^2 k_B}B^\\prime_\\nu(T_{\\rm CMB, 0})

    Here we calculate the RHS of this equation and compare it to the
    ratio of T_RJ and the result of its transformation to T_CMB.
    """
    T_CMB, T_RJ, freqs, test_map_RJ, test_map_CMB, test_map_dimless = setUp
    equiv = {"equivalencies": units.cmb_equivalencies(freqs)}
    rj_from_cmb = T_CMB.to(units.K_RJ, **equiv)
    cmb_from_rj = T_RJ.to(units.K_CMB, **equiv)

    # check that the reverse transformation gives overall transformation of unity.
    reverse1 = rj_from_cmb.to(units.K_CMB, **equiv)
    reverse2 = cmb_from_rj.to(units.K_RJ, **equiv)

    np.testing.assert_almost_equal(1.0, T_CMB / reverse1, decimal=6)
    np.testing.assert_almost_equal(1.0, T_RJ / reverse2, decimal=6)


def test_fits_unit_functionality(setUp, tmp_path):
    """Test that the units can be written to the fits header. Check that
    they can be read in again and assigned to the data in that fits file
    correctly.
    """
    T_CMB, T_RJ, freqs, test_map_RJ, test_map_CMB, test_map_dimless = setUp
    hp.write_map(
        tmp_path / "temp_fits_file_RJ.fits",
        test_map_RJ.value,
        column_units=test_map_RJ.unit.to_string("generic"),
    )
    hp.write_map(
        tmp_path / "temp_fits_file_CMB.fits",
        test_map_CMB.value,
        column_units=test_map_CMB.unit.to_string("generic"),
    )
    hp.write_map(
        tmp_path / "temp_fits_file_dimless.fits",
        test_map_dimless.value,
        column_units=test_map_dimless.unit.to_string("generic"),
    )
    hp.write_map(tmp_path / "temp_fits_file_no_unit_hdr.fits", test_map_dimless.value)

    cmb_in = read_map(tmp_path / "temp_fits_file_CMB.fits", 256)
    rj_in = read_map(tmp_path / "temp_fits_file_RJ.fits", 256)
    dimless_in = read_map(tmp_path / "temp_fits_file_dimless.fits", 256)
    no_unit_hdr = read_map(tmp_path / "temp_fits_file_no_unit_hdr.fits", 256)
    assert cmb_in.unit == units.K_CMB
    assert rj_in.unit == units.K_RJ
    assert dimless_in.unit == units.dimensionless_unscaled
    assert no_unit_hdr.unit == units.dimensionless_unscaled
