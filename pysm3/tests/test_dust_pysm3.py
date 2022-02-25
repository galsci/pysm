from astropy.tests.helper import assert_quantity_allclose

# import healpy as hp
# from pysm3.models.dust import blackbody_ratio

import pysm3
from pysm3 import units as u

import pytest


@pytest.mark.parametrize("model_tag", ["d11"])
def test_dust_model_353(model_tag):
    nside = 2048

    freq = 353 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        "dust_gnilc/gnilc_dust_template_nside{nside}.fits".format(nside=nside),
        nside=nside,
        field=(0, 1, 2),
    )
    rtol = 1e-5

    # if model_tag == "d11":
    #    beam = 1 * u.deg
    #    input_template = hp.smoothing(input_template, fwhm=beam.to_value(u.radians))
    #    output = hp.smoothing(output, fwhm=beam.to_value(u.radians))
    #    rtol = 1e-2

    assert_quantity_allclose(input_template, output, rtol=rtol)


def test_d9_vs_d11():
    nside = 2048

    freq = 857 * u.GHz

    output_d9 = pysm3.Sky(preset_strings=["d9"], nside=nside).get_emission(freq)
    output_d11 = pysm3.Sky(preset_strings=["d11"], nside=nside).get_emission(freq)

    rtol = 1e-5

    assert_quantity_allclose(output_d9, output_d11, rtol=rtol)


# @pytest.mark.parametrize("model_tag", ["d9", "d10"])
# def test_gnilc_857(model_tag):
#    freq = 857 * u.GHz
#
#    model = pysm3.Sky(preset_strings=[model_tag], nside=2048)
#
#    output = model.get_emission(freq)
#
#    input_template = pysm3.models.read_map(
#        "dust_gnilc/gnilc_dust_template_nside{nside}.fits".format(nside=2048),
#        nside=2048,
#        field=(0, 1, 2),
#    )
#
#    freq_ref = 353 * u.GHz
#    beta = 1.48 if model_tag == "d9" else pysm3.models.read_map(
#        "dust_gnilc/gnilc_dust_beta_nside{nside}.fits".format(nside=2048),
#        nside=2048,
#        field=0,
#    )
#    Td = 19.6 * u.K if model_tag == "d9" else pysm3.models.read_map(
#        "dust_gnilc/gnilc_dust_Td_nside{nside}.fits".format(nside=2048),
#        nside=2048,
#        field=0,
#    )
#    scaling = (freq / freq_ref) ** (beta - 2)
#    scaling *= blackbody_ratio(freq, freq_ref, Td.to_value(u.K))
#
#    assert_quantity_allclose(input_template * scaling, output, rtol=1e-6)
