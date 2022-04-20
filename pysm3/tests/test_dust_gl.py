import psutil
from astropy.tests.helper import assert_quantity_allclose

from pysm3.models.dust import blackbody_ratio

import pysm3
from pysm3 import units as u

import pytest


def test_d11gl_lownside():
    nside = 64

    freq = 857 * u.GHz

    output = pysm3.Sky(preset_strings=["d11gl"], nside=nside).get_emission(freq)


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
