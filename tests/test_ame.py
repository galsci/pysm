import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3


@pytest.mark.parametrize("freq", [30, 100])
@pytest.mark.parametrize("model", ["a1"])
def test_model(model, freq):

    model = pysm3.Sky(preset_strings=[model], nside=64)

    model_number = 3
    expected_map = pysm3.read_map(
        f"pysm_2_test_data/check{model_number}spinn_{freq}p0_64.fits",
        64,
        unit=pysm3.units.uK_RJ,
        field=0,
    )

    emission = model.get_emission(freq << pysm3.units.GHz)
    assert_quantity_allclose(expected_map, emission[0], rtol=1e-5)

    for i in [1, 2]:
        assert_quantity_allclose(0 * pysm3.units.uK_RJ, emission[i])
