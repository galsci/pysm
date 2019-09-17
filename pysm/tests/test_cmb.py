import numpy as np
import pysm
import pysm.units as u
from astropy.tests.helper import assert_quantity_allclose


def test_cmb_map():

    nside = 32

    # pretend for testing that the Dust is CMB
    model = pysm.CMBMap(map_IQU="pysm_2/lensed_cmb.fits", nside=nside)

    freq = 100 * u.GHz

    expected_map = pysm.read_map(
        "pysm_2/lensed_cmb.fits", field=(0, 1), nside=nside, unit=u.uK_CMB
    ).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq))

    simulated_map = model.get_emission(freq)
    for pol in [0, 1]:
        assert_quantity_allclose(expected_map[pol], simulated_map[pol], rtol=1e-5)


def test_cmb_map_bandpass():

    nside = 32

    # pretend for testing that the Dust is CMB
    model = pysm.CMBMap(map_IQU="pysm_2/lensed_cmb.fits", nside=nside)

    freq = 100 * u.GHz

    expected_map = pysm.read_map(
        "pysm_2/lensed_cmb.fits", field=0, nside=nside, unit=u.uK_CMB
    ).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq))

    print(
        "expected_scaling",
        (1 * u.K_CMB).to_value(u.K_RJ, equivalencies=u.cmb_equivalencies(freq)),
    )

    freqs = np.array([98, 99, 100, 101, 102]) * u.GHz
    weights = np.ones(len(freqs))

    # just checking that the result is reasonably close to the delta frequency at the center frequency

    assert_quantity_allclose(
        expected_map, model.get_emission(freqs, weights)[0], rtol=1e-3
    )
