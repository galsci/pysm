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


def test_dipole_quadrupole_correction():
    nside = 64
    amp = 3366.6 * u.uK_CMB
    T_cmb = 2.7255 * u.K_CMB
    dip_lon = 263.986 * u.deg
    dip_lat = 48.247 * u.deg

    # Standard dipole
    dipole = CMBDipole(nside, amp, T_cmb, dip_lon, dip_lat)
    dipole_map = dipole.get_emission(100 * u.GHz)

    # Dipole with quadrupole correction
    dipole_quad = CMBDipole(
        nside, amp, T_cmb, dip_lon, dip_lat, quadrupole_correction=True
    )
    dipole_quad_map = dipole_quad.get_emission(100 * u.GHz)

    # The maps should not be identical
    assert not np.allclose(dipole_map.value, dipole_quad_map.value)

    # The quadrupole-corrected map should differ by a small but nonzero amount
    diff = np.abs(dipole_map.value - dipole_quad_map.value)
    assert np.any(diff > 0)
    # The difference should be small compared to the dipole amplitude
    assert np.max(diff) < 1e-2 * np.max(np.abs(dipole_map.value))

    # Spherical harmonics transform
    lmax = 3
    alm_dipole = hp.map2alm(dipole_map.value, lmax=lmax)
    alm_quad = hp.map2alm(dipole_quad_map.value, lmax=lmax)

    # Compute power in dipole (l=1) and quadrupole (l=2) for both maps
    cl_dipole = hp.alm2cl(alm_dipole)
    cl_quad = hp.alm2cl(alm_quad)

    # Compare dipole (l=1) and quadrupole (l=2) components
    # l=1 is index 1, l=2 is index 2
    dipole_power = cl_dipole[1]
    quad_power = cl_dipole[2]
    dipole_quad_power = cl_quad[1]
    quad_quad_power = cl_quad[2]

    # Dipole power should be similar for both
    np.testing.assert_allclose(dipole_power, dipole_quad_power, rtol=1e-3)

    # Quadrupole power should be larger for the quadrupole-corrected map
    assert 1.5 * quad_power < quad_quad_power < 2 * quad_power


def test_dipole_preset_dip2_quadrupole():
    import pysm3

    nside = 64
    sky1 = pysm3.Sky(nside=nside, preset_strings=["dip1"], output_unit=u.uK_CMB)
    sky2 = pysm3.Sky(nside=nside, preset_strings=["dip2"], output_unit=u.uK_CMB)

    dipole_map1 = sky1.get_emission([80, 90, 100, 110] * u.GHz)
    dipole_map2 = sky2.get_emission([80, 90, 100, 110] * u.GHz)

    # Spherical harmonics transform
    lmax = 3
    alm1 = hp.map2alm(dipole_map1.value, lmax=lmax)
    alm2 = hp.map2alm(dipole_map2.value, lmax=lmax)

    cl1 = hp.alm2cl(alm1)
    cl2 = hp.alm2cl(alm2)

    # Dipole power should be similar
    np.testing.assert_allclose(cl1[1], cl2[1], rtol=1e-3)
    # Quadrupole power should be larger for dip2
    assert cl2[2] > cl1[2]


if __name__ == "__main__":
    test_dipole_fit()
    test_dipole_fit_with_sky()
    test_dipole_quadrupole_correction()
    test_dipole_preset_dip2_quadrupole()
