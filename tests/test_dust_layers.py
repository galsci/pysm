import healpy as hp
import numpy as np
import pytest
from astropy import units
from astropy.tests.helper import assert_quantity_allclose

from pysm3 import Sky, read_map
from pysm3 import units as u
from pysm3.models.dust_layers import ModifiedBlackBodyLayers


def test_modified_black_body_class():
    num_layers = 3
    nside = 8
    npix = hp.nside2npix(nside)

    # Create 3 layers at 1, 2 and 3 uK_RJ
    map_layers = np.ones((num_layers, 3, npix), dtype=np.float64)
    for i in range(num_layers):
        map_layers[i] *= i + 1

    model = ModifiedBlackBodyLayers(
        map_layers,
        freq_ref=353 * u.GHz,
        map_mbb_index=np.ones((num_layers, npix), dtype=np.float64),
        map_mbb_temperature=np.ones((num_layers, npix), dtype=np.float64),
        nside=nside,
        num_layers=num_layers,
        unit_layers=u.uK_RJ,
        unit_mbb_temperature=u.K,
        map_dist=None,
    )

    # At the reference frequency of 353GHz, the output should be 6 uK_RJ
    expected_output = u.Quantity(
        np.sum(np.arange(1, num_layers + 1)) * np.ones((3, npix), dtype=np.float64),
        unit=u.uK_RJ,
    )
    freq = 353
    rtol = 1e-7
    assert_quantity_allclose(
        expected_output, model.get_emission(freq * units.GHz), rtol=rtol
    )


@pytest.mark.parametrize("freq", [100, 353, 857])
def test_model_d12(freq):
    sky = Sky(
        preset_strings=["d12"],
        nside=8,
        output_unit=u.MJy / u.sr,
    )

    emission = sky.get_emission(freq * u.GHz)

    expected_map = (
        read_map(
            f"mkd_dust/test/layermodel_nside8_{freq}.fits",
            8,
            unit=u.MJy / u.sr,
            field=(0, 1, 2),
        )
        * 0.911
    )

    assert_quantity_allclose(expected_map, emission, rtol=1e-5)
