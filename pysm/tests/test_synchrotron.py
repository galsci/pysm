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
    ).reshape((1, 3, -1)).value

    np.testing.assert_allclose(
        synch, synchrotron.get_emission(freq), rtol=1e-5
    )
