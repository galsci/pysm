import pysm

import pytest
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model", ["s1"])
# @pytest.mark.parametrize("model", ["s1", "s2", "s3"]) # FIXME activate testing for other models
def test_synchrotron_model(model, freq):

    synchrotron = pysm.Sky(preset_strings=[model], nside=64)

    model_number = {"s1": 2, "s2": 7, "s3": 10}[model]
    synch = pysm.read_map(
        "pysm_2_test_data/check{}synch_{}p0_64.fits".format(model_number, freq),
        64,
        unit=pysm.units.uK_RJ,
        field=(0, 1, 2),
    )

    assert_quantity_allclose(
        synch, synchrotron.get_emission(freq << pysm.units.GHz), rtol=1e-5
    )
