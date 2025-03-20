import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3
import pysm3.units as u


def test_cmb_map():

    nside = 32

    model = pysm3.CMBMap(map_IQU="pysm_2/lensed_cmb.fits", nside=nside)

    freq = 100 * u.GHz

    expected_map = pysm3.read_map(
        "pysm_2/lensed_cmb.fits", field=(0, 1), nside=nside, unit=u.uK_CMB
    ).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq))

    simulated_map = model.get_emission(freq)
    for pol in [0, 1]:
        assert_quantity_allclose(expected_map[pol], simulated_map[pol], rtol=1e-5)


def test_cmb_map_bandpass():

    nside = 32

    # pretend for testing that the Dust is CMB
    model = pysm3.CMBMap(map_IQU="pysm_2/lensed_cmb.fits", nside=nside)

    freq = 100 * u.GHz

    expected_map = pysm3.read_map(
        "pysm_2/lensed_cmb.fits", field=0, nside=nside, unit=u.uK_CMB
    ).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq))

    print(
        "expected_scaling",
        (1 * u.K_CMB).to_value(u.K_RJ, equivalencies=u.cmb_equivalencies(freq)),
    )

    freqs = np.array([98, 99, 100, 101, 102]) * u.GHz
    weights = np.ones(len(freqs))

    # just checking that the result is reasonably close
    # to the delta frequency at the center frequency

    assert_quantity_allclose(
        expected_map, model.get_emission(freqs, weights)[0], rtol=1e-3
    )


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model_tag", ["c1"])
def test_cmb_lensed(model_tag, freq):

    # The PySM test was done with a different seed than the one
    # baked into the preset models
    pysm3.sky.PRESET_MODELS["c1"]["cmb_seed"] = 1234
    model = pysm3.Sky(preset_strings=[model_tag], nside=64)

    model_number = 5
    expected_output = pysm3.read_map(
        f"pysm_2_test_data/check{model_number}cmb_{freq}p0_64.fits",
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    assert_quantity_allclose(
        expected_output, model.get_emission(freq * u.GHz), rtol=1e-5
    )


def test_cmb_lensed_no_delens():

    # Addressing issue #213: CMBLensed model was raising a ValueError when
    # apply_delens=True. This first confirms that the model works without
    # apply_delens=True
    model  = pysm3.models.CMBLensed(
        nside=64, 
        cmb_spectra='pysm_2/camb_lenspotentialCls.dat',
        apply_delens=False,
        delensing_ells='pysm_2/delens_ells.txt'
    )

def test_cmb_lensed_delens():
    # Further addressing issue #213: CMBLensed model raised a ValueError
    # when apply_delens=True. This confirms it has been resolved.

    model  = pysm3.models.CMBLensed(
        nside=64, 
        cmb_spectra='pysm_2/camb_lenspotentialCls.dat',
        apply_delens=True,
        delensing_ells='pysm_2/delens_ells.txt'
    )
