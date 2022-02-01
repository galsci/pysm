import astropy.units as units
import numpy as np
import healpy as hp
from astropy.tests.helper import assert_quantity_allclose

import pysm3
from pysm3 import units as u
import pytest


@pytest.mark.parametrize("model_tag", ["d9"])
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
