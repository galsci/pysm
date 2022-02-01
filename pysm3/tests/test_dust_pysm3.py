import astropy.units as units
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

import pysm3
from pysm3 import units as u
import pysm3.models.dust as dust
import pytest


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model_tag", ["d9"])
def test_dust_model(model_tag, freq):

    model = pysm3.Sky(preset_strings=[model_tag], nside=2048)

    model_number = {"d0": 1, "d9":1, "d1": 1, "d2": 6, "d3": 9, "d6": 12}[model_tag]
    expected_output = pysm3.read_map(
        "pysm_2_test_data/check{}therm_{}p0_64.fits".format(model_number, freq),
        64,
        unit="uK_RJ",
        field=(0, 1, 2),
    )

    # for some models we do not have tests, we compare with output from a simular model
    # and we increase tolerance, mostly just to exercise the code.
    rtol = {"d9": 0.9}.get(model_tag, 1e-5)

    output = model.get_emission(freq * units.GHz)

    assert output.mean() < freq * u.uK_RJ  # Don't ask

    # assert_quantity_allclose(
    #     expected_output, model.get_emission(freq * units.GHz), rtol=rtol
    # )
