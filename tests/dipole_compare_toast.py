import pytest
import numpy as np
import healpy as hp
from pysm3 import units as u
from pysm3.models.dipole import CMBDipole, CMBDipoleQuad
from pysm3.models.template import read_map

# Reference data
NSIDE = 256
FREQUENCIES = [0, 80, 90, 100, 110] * u.GHz
REFERENCE_URLS = {
    0 * u.GHz: "test_data/dipole/dipole_nside0256_freq000GHz.fits",
    80 * u.GHz: "test_data/dipole/dipole_nside0256_freq080GHz.fits",
    90 * u.GHz: "test_data/dipole/dipole_nside0256_freq090GHz.fits",
    100 * u.GHz: "test_data/dipole/dipole_nside0256_freq100GHz.fits",
    110 * u.GHz: "test_data/dipole/dipole_nside0256_freq110GHz.fits",
}

# Default parameters from presets.cfg
DIPOLE_AMP = "3366.6 uK_CMB"
T_CMB = "2.7255 K_CMB"
DIPOLE_LON = "263.986 deg"
DIPOLE_LAT = "48.247 deg"

@pytest.mark.parametrize("freq", [80, 90, 100, 110] * u.GHz)
def test_quadrupole_corrected_freqs(freq):
    # Load reference map
    reference_map = read_map(REFERENCE_URLS[freq], nside=NSIDE)

    # Instantiate CMBDipoleQuad (quadrupole correction enabled)
    dipole_model = CMBDipoleQuad(
        nside=NSIDE,
        amp=DIPOLE_AMP,
        T_cmb=T_CMB,
        dip_lon=DIPOLE_LON,
        dip_lat=DIPOLE_LAT,
    )

    # Generate dipole map
    generated_map = dipole_model.get_emission(freq)

    # Convert generated map to K_CMB for comparison
    generated_map_K_CMB = generated_map.to(u.K_CMB, equivalencies=u.cmb_equivalencies(freq))

    # Compare maps
    np.testing.assert_allclose(
        generated_map_K_CMB.value, reference_map.value, rtol=1e-6, atol=1e-6
    )

def test_no_quadrupole():
    # Load 0 GHz reference map
    ref_0ghz_map = read_map(REFERENCE_URLS[0 * u.GHz], nside=NSIDE)

    # Instantiate CMBDipole (no quadrupole correction)
    dipole_model = CMBDipole(
        nside=NSIDE,
        amp=DIPOLE_AMP,
        T_cmb=T_CMB,
        dip_lon=DIPOLE_LON,
        dip_lat=DIPOLE_LAT,
        quadrupole_correction=False,
    )

    # Generate map at 100 GHz with no quadrupole correction
    generated_100ghz_map = dipole_model.get_emission(100 * u.GHz)

    # Convert generated map to K_CMB for comparison
    generated_100ghz_map_K_CMB = generated_100ghz_map.to(u.K_CMB, equivalencies=u.cmb_equivalencies(100 * u.GHz))

    # Compare maps
    # As discussed, this comparison is physically questionable, but requested.
    # It will likely fail due to physical differences between a 100 GHz map
    # and a 0 GHz reference map.
    np.testing.assert_allclose(
        generated_100ghz_map_K_CMB.value, ref_0ghz_map.value, rtol=1e-6, atol=1e-6
    )