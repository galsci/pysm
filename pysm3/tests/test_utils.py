import numpy as np
import pysm3
import pysm3.units as u
from astropy.tests.helper import assert_quantity_allclose


def test_has_polarization():
    h = pysm3.utils.has_polarization

    m = np.empty(12)
    assert h(np.empty((3, 12)))
    assert not h(np.empty((1, 12)))
    assert not h(m)
    assert h(np.empty((4, 3, 12)))
    assert not h(np.empty((4, 1, 12)))
    assert h((m, m, m))
    assert h([(m, m, m), (m, m, m)])


def test_bandpass_unit_conversion():
    nside = 32
    freqs = np.array([250, 300, 350]) * u.GHz
    weights = np.ones(len(freqs))
    sky = pysm3.Sky(nside=nside, preset_strings=["c2"])
    CMB_rj_int = sky.get_emission(freqs, weights)
    CMB_thermo_int = CMB_rj_int*pysm3.utils.bandpass_unit_conversion(
        freqs, weights, u.uK_CMB
    )
    expected_map = pysm3.read_map(
        "pysm_2/lensed_cmb.fits", field=(0, 1), nside=nside, unit=u.uK_CMB
    )
    for pol in [0, 1]:
        assert_quantity_allclose(expected_map[pol], CMB_thermo_int[pol], rtol=1e-4)
