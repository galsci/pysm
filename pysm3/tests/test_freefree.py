import pysm3

import pytest
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("freq", [30, 100, 353])
@pytest.mark.parametrize("model", ["f1"])
def test_model(model, freq):

    model = pysm3.Sky(preset_strings=[model], nside=64)

    model_number = 4
    expected_map = pysm3.read_map(
        "pysm_2_test_data/check{}freef_{}p0_64.fits".format(model_number, freq),
        64,
        unit=pysm3.units.uK_RJ,
        field=0,
    )

    assert_quantity_allclose(
        expected_map, model.get_emission(freq << pysm3.units.GHz)[0], rtol=1e-5
    )
