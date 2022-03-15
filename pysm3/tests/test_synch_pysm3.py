import psutil
from astropy.tests.helper import assert_quantity_allclose

from pysm3.models.dust import blackbody_ratio

import pysm3
from pysm3 import units as u

import pytest


@pytest.mark.parametrize("model_tag", ["s4", "s5"])
def test_synch_model_noscaling(model_tag):
    nside = 2048

    freq = 23 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        "synch/synch_template_nside{nside}.fits".format(nside=nside),
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
        "synch/synch_template_nside{nside}.fits".format(nside=nside),
        nside=nside,
        field=(0, 1, 2),
    )

    freq_ref = 23 * u.GHz
    beta = (
        -3.1
        if model_tag == "s4"
        else pysm3.models.read_map(
            "synch/synch_beta_nside{nside}.fits".format(nside=nside),
            nside=nside,
            field=0,
        )
    )
    scaling = (freq / freq_ref) ** beta

    assert_quantity_allclose(input_template * scaling, output, rtol=1e-6)


# @pytest.mark.skipif(
#     psutil.virtual_memory().total * u.byte < 20 * u.GB,
#     reason="Running d11 at high lmax requires 20 GB of RAM",
# )
# def test_d10_vs_d11():
#     nside = 2048
#
#     freq = 857 * u.GHz
#
#     output_d10 = pysm3.Sky(preset_strings=["d10"], nside=nside).get_emission(freq)
#     d11_configuration = pysm3.sky.PRESET_MODELS["d11"].copy()
#     del d11_configuration["class"]
#     d11 = pysm3.models.ModifiedBlackBodyRealization(
#         nside=nside, seeds=[8192, 777, 888], synalm_lmax=16384, **d11_configuration
#     )
#     output_d11 = d11.get_emission(freq)
#
#     rtol = 1e-5
#
#     assert_quantity_allclose(output_d10, output_d11, rtol=rtol, atol=0.05 * u.uK_RJ)
