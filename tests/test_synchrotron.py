import pysm3

import pytest
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model", ["s0", "s1", "s2", "s3"])
def test_synchrotron_model(model, freq):

    synchrotron = pysm3.Sky(preset_strings=[model], nside=64)

    model_number = {"s0": 2, "s1": 2, "s2": 7, "s3": 10}[model]
    synch = pysm3.read_map(
        "pysm_2_test_data/check{}synch_{}p0_64.fits".format(model_number, freq),
        64,
        unit=pysm3.units.uK_RJ,
        field=(0, 1, 2),
    )

    # for some models we do not have tests, we compare with output from a simular model
    # and we increase tolerance, mostly just to exercise the code.
    rtol = {"s0": 5}.get(model, 1e-5)

    assert_quantity_allclose(
        synch, synchrotron.get_emission(freq << pysm3.units.GHz), rtol=rtol
    )
