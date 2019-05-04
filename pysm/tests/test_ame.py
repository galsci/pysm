import pysm

import pytest
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("freq", [30, 100])
@pytest.mark.parametrize("model", ["a1"])
def test_model(model, freq):

    model = pysm.preset_models(model, nside=64)

    model_number = 3
    expected_map = pysm.read_map(
        "pysm_2_test_data/check{}spinn_{}p0_64.fits".format(model_number, freq),
        64,
        unit=pysm.units.uK_RJ,
        field=0,
    ).reshape((1, 1, -1))

    assert_quantity_allclose(
        expected_map, model.get_emission(freq << pysm.units.GHz), rtol=1e-5
    )
