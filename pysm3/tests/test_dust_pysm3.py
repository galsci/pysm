from astropy.tests.helper import assert_quantity_allclose
from pysm3.models.dust import blackbody_ratio

import pysm3
from pysm3 import units as u
import pytest


@pytest.mark.parametrize("model_tag", ["d11"])
def test_dust_model_353(model_tag):
    freq = 353 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=2048)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        "dust_gnilc/gnilc_dust_template_nside{nside}.fits".format(nside=2048),
        nside=2048,
        field=(0, 1, 2),
    )

    assert_quantity_allclose(input_template, output)


#@pytest.mark.parametrize("model_tag", ["d9", "d10"])
#def test_gnilc_857(model_tag):
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
