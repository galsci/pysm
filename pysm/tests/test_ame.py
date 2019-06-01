import pysm

import pytest
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("freq", [30, 100])
@pytest.mark.parametrize("model", ["a1"])
def test_model(model, freq):

    model = pysm.Sky(preset_strings=[model], nside=64)

    model_number = 3
    expected_map = pysm.read_map(
        "pysm_2_test_data/check{}spinn_{}p0_64.fits".format(model_number, freq),
        64,
        unit=pysm.units.uK_RJ,
        field=0,
    )

    emission = model.get_emission(freq << pysm.units.GHz)
    assert_quantity_allclose(expected_map, emission[0], rtol=1e-5)

    for i in [1, 2]:
        assert_quantity_allclose(0 * pysm.units.uK_RJ, emission[i])
