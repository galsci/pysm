import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3


@pytest.mark.parametrize("freq", [30, 100])
@pytest.mark.parametrize("model", ["a2"])
def test_model(model, freq):

    model = pysm3.Sky(preset_strings=[model], nside=64)

    model_number = 8
    expected_map = pysm3.read_map(
        f"pysm_2_test_data/check{model_number}spinn_{freq}p0_64.fits",
        64,
        unit=pysm3.units.uK_RJ,
        field=(0, 1, 2),
    )

    assert_quantity_allclose(
        expected_map, model.get_emission(freq << pysm3.units.GHz), rtol=1e-3
    )
