import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from unittest.mock import patch

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


@pytest.mark.parametrize("freq", [100])
def test_cmb_lensed_with_patch_object(freq):
    # The CMBLensed Model expects the CAMB power spectrum output to 
    # begin at l=2, resulting misalignment of ells when it is not 
    # included. To test the fix, the method must be patched.
    # This test shows that patching works, complementing test_cmb_lensed_l01.

    model_number = 5
    expected_output = pysm3.read_map(
        f"pysm_2_test_data/check{model_number}cmb_{freq}p0_64.fits",
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    # Use a temporary model to get access to the read_txt method
    temp_model = pysm3.Model(nside=64)
    c1_spectra = temp_model.read_txt('pysm_2/camb_lenspotentialCls.dat',
                                     unpack=True)
    c1_delens_l = temp_model.read_txt('pysm_2/delens_ells.txt',
                                      unpack=True)

    with patch.object(target=pysm3.models.cmb.CMBLensed, 
                      attribute="read_txt", 
                      side_effect=[c1_spectra, c1_delens_l]):
        # The PySM test was done with a different seed than the one
        # baked into the preset models
        c1 = pysm3.models.cmb.CMBLensed(
            nside=64,
            cmb_spectra='pysm_2/camb_lenspotentialCls.dat',
            cmb_seed=1234
        )
        model = pysm3.Sky(nside=64, component_objects=[c1])

        assert_quantity_allclose(
            expected_output, 
            model.get_emission(freq * u.GHz), 
            rtol=1e-5
        )


@pytest.mark.parametrize("freq", [100])
def test_cmb_lensed_l01(freq):
    # The CMBLensed Model expects the CAMB power spectrum output to 
    # begin at l=2, resulting misalignment of ells when it is not 
    # included. This test shows that the fix works, by changing the 
    # input cls from the original test (test_cmb_lensed) to include l=0,1.

    model_number = 5
    expected_output = pysm3.read_map(
        f"pysm_2_test_data/check{model_number}cmb_{freq}p0_64.fits",
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    # Use a temporary model to get access to the read_txt method
    temp_model = pysm3.Model(nside=64)
    c1_spectra = temp_model.read_txt('pysm_2/camb_lenspotentialCls.dat',
                                     unpack=True)
    c1_delens_l = temp_model.read_txt('pysm_2/delens_ells.txt',
                                      unpack=True)

    # Add rows of 0s for l=0,1
    c1_spectra_l01 = np.zeros((c1_spectra.shape[0], c1_spectra.shape[1] + 2))
    c1_spectra_l01[:, 2:] = c1_spectra

    with patch.object(target=pysm3.models.cmb.CMBLensed,
                      attribute="read_txt",
                      side_effect=[c1_spectra_l01, c1_delens_l]):
        c1 = pysm3.models.cmb.CMBLensed(
            nside=64,
            cmb_spectra='pysm_2/camb_lenspotentialCls.dat',
            cmb_seed=1234
        )
        model = pysm3.Sky(nside=64, component_objects=[c1])

        assert_quantity_allclose(
            expected_output, 
            model.get_emission(freq * u.GHz), 
            rtol=1e-5
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

def test_cmb_map_inmemory():
    nside = 32
    npix = 12 * nside ** 2
    # Create a fake IQU map with astropy units
    arr = np.ones((3, npix)) * 42
    arr = arr * u.uK_CMB
    model = pysm3.CMBMap(map_IQU=arr, nside=nside)
    freq = 100 * u.GHz
    # Should return the same map (scaled to uK_RJ)
    simulated_map = model.get_emission(freq)
    expected = arr.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq))
    assert_quantity_allclose(simulated_map, expected)


def test_cmb_map_inmemory_no_units():
    nside = 32
    npix = 12 * nside ** 2
    arr = np.ones((3, npix)) * 42  # No units
    with pytest.raises(ValueError, match="astropy units"):
        pysm3.CMBMap(map_IQU=arr, nside=nside)


def test_cmb_map_inmemory_wrong_shape():
    nside = 32
    npix = 12 * nside ** 2
    arr = np.ones((2, npix)) * 42 * u.uK_CMB  # Wrong shape: should be (1, npix) or (3, npix)
    with pytest.raises(ValueError, match="shape"):
        pysm3.CMBMap(map_IQU=arr, nside=nside)
