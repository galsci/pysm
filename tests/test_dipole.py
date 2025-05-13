import numpy as np
import healpy as hp
import pysm3.units as u
from astropy.constants import c
from pysm3.models.dipole import CMBDipole
from astropy.tests.helper import assert_quantity_allclose


def test_dipole_fit():
    nside = 64
    amp = 3366.6 * u.uK_CMB
    T_cmb = 2.7255 * u.K_CMB
    dip_lon = 263.986 * u.deg
    dip_lat = 48.247 * u.deg

    dipole = CMBDipole(nside, amp, T_cmb, dip_lon, dip_lat)
    dipole_map = dipole.get_emission(100 * u.GHz)

    dipole_map = dipole_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(100 * u.GHz))

    # Fit dipole using healpy
    _, dip_vec = hp.fit_dipole(dipole_map.value)
    fit_amp = np.linalg.norm(dip_vec) * dipole_map.unit

    lon, lat = hp.vec2dir(dip_vec, lonlat=True)

    # Adjust longitude to [0, 360) for comparison
    lon = lon % 360

    # Check amplitude within 0.1%
    assert_quantity_allclose(amp, fit_amp, rtol=1e-3)

    # Check direction within 0.01 deg
    np.testing.assert_allclose(lon, dip_lon.value, atol=0.01)
    np.testing.assert_allclose(lat, dip_lat.value, atol=0.01)


def test_dipole_fit_with_sky():
    import pysm3
    from astropy.tests.helper import assert_quantity_allclose

    # Use the dip1 preset, which uses the dipole component
    sky = pysm3.Sky(nside=64, preset_strings=["dip1"], output_unit=u.uK_CMB)
    # Evaluate at 100 GHz
    dipole_map = sky.get_emission(100 * u.GHz)

    # Fit dipole using healpy
    _, dip_vec = hp.fit_dipole(dipole_map.value)
    fit_amp = np.linalg.norm(dip_vec) * dipole_map.unit

    # Get preset parameters for dipole
    amp = 3366.6 * u.uK_CMB
    dip_lon = 263.986
    dip_lat = 48.247

    lon, lat = hp.vec2dir(dip_vec, lonlat=True)
    lon = lon % 360

    # Check amplitude within 0.1%
    assert_quantity_allclose(amp, fit_amp, rtol=1e-3)
    # Check direction within 0.01 deg
    np.testing.assert_allclose(lon, dip_lon, atol=0.01)
    np.testing.assert_allclose(lat, dip_lat, atol=0.01)


if __name__ == "__main__":
    test_dipole_fit()
    test_dipole_fit_with_sky()
