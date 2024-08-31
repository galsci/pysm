import numpy as np
import psutil
import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3
from pysm3 import units as u


@pytest.mark.parametrize("model_tag", ["s7"])
def test_synch_model_s7_noscaling(model_tag):
    nside = 2048

    freq = 23 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"synch/synch_template_nside{nside}_2023.02.25.fits",
        nside=nside,
        field=(0, 1, 2),
    )
    rtol = 1e-5

    assert_quantity_allclose(input_template, output, rtol=rtol)


@pytest.mark.parametrize("model_tag", ["s7"])
def test_synch_model_s7_44(model_tag):
    nside = 2048

    freq = 44 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"synch/synch_template_nside{nside}_2023.02.25.fits",
        nside=nside,
        field=(0, 1, 2),
    )

    freq_ref = 23 * u.GHz
    beta = pysm3.models.read_map(
        f"synch/synch_beta_nside{nside}_2023.02.16.fits",
        nside=nside,
        field=0,
    )
    curvature = pysm3.models.read_map(
        f"synch/synch_curvature_nside{nside}_2023.02.17.fits",
        nside=nside,
        field=0,
    )
    curvature_term = np.log((freq / (23 * u.GHz)) ** curvature)
    scaling = (freq / freq_ref) ** (beta + curvature_term)

    assert_quantity_allclose(input_template * scaling, output, rtol=1e-6)


@pytest.mark.parametrize("model_tag", ["s4", "s5"])
def test_synch_model_noscaling(model_tag):
    nside = 2048

    freq = 23 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"synch/synch_template_nside{nside}_2023.02.25.fits",
        nside=nside,
        field=(0, 1, 2),
    )
    rtol = 1e-5

    assert_quantity_allclose(input_template, output, rtol=rtol)


@pytest.mark.parametrize("model_tag", ["s4", "s5"])
def test_synch_44(model_tag):
    freq = 44 * u.GHz
    nside = 2048

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"synch/synch_template_nside{nside}_2023.02.25.fits",
        nside=nside,
        field=(0, 1, 2),
    )

    freq_ref = 23 * u.GHz
    beta = (
        -3.1
        if model_tag == "s4"
        else pysm3.models.read_map(
            f"synch/synch_beta_nside{nside}_2023.02.16.fits",
            nside=nside,
            field=0,
        )
    )
    scaling = (freq / freq_ref) ** beta

    assert_quantity_allclose(input_template * scaling, output, rtol=1e-6)


@pytest.mark.skipif(
    psutil.virtual_memory().total * u.byte < 20 * u.GB,
    reason="Running s6 at high lmax requires 20 GB of RAM",
)
@pytest.mark.parametrize("freq", [23, 44])
def test_s6_vs_s5(freq):
    nside = 2048

    freq = freq * u.GHz

    output_s5 = pysm3.Sky(preset_strings=["s5"], nside=nside).get_emission(freq)
    s6_configuration = pysm3.sky.PRESET_MODELS["s6"].copy()
    del s6_configuration["class"]
    s6 = pysm3.models.PowerLawRealization(
        nside=nside, synalm_lmax=16384, seeds=[555, 444], **s6_configuration
    )
    output_s6 = s6.get_emission(freq)

    rtol = 1e-5

    assert_quantity_allclose(output_s5, output_s6, rtol=rtol, atol=0.05 * u.uK_RJ)
