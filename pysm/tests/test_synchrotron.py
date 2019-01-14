import unittest
import numpy as np
import pysm

import pytest


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model", ["s1"])
# @pytest.mark.parametrize("model", ["s1", "s2", "s3"]) # FIXME activate testing for other models
def test_synchrotron_model(model, freq):

    synchrotron = pysm.preset_models(model, nside=64)

    model_number = {"s1": 2, "s2": 7, "s3": 10}[model]
    synch = pysm.read_map(
        "pysm_2_test_data/check{}synch_{}p0_64.fits".format(model_number, freq),
        64,
        field=(0, 1, 2),
    ).value

    model_frac_diff = (synch - synchrotron.get_emission(freq * pysm.units.GHz)) / synch

    np.testing.assert_array_almost_equal(
        model_frac_diff, np.zeros_like(model_frac_diff), decimal=6
    )
