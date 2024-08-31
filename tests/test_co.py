import healpy as hp
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from pysm3 import COLines, Sky
from pysm3 import units as u
from pysm3.utils import RemoteData


@pytest.mark.parametrize("include_high_galactic_latitude_clouds", [False, True])
def test_co(include_high_galactic_latitude_clouds):

    line = "10"

    co = COLines(
        nside=16,
        has_polarization=True,
        lines=[line],
        include_high_galactic_latitude_clouds=include_high_galactic_latitude_clouds,
        polarization_fraction=0.001,
        theta_high_galactic_latitude_deg=20.0,
        random_seed=1234567,
        verbose=False,
        run_mcmole3d=False,
    )

    line_freq = co.line_frequency[line]
    co_map = co.get_emission(line_freq)

    tag = "wHGL" if include_high_galactic_latitude_clouds else "noHGL"
    remote_data = RemoteData()
    expected_map_filename = remote_data.get(
        f"co/testing/CO10_TQUmaps_{tag}_nside16_ring.fits.zip"
    )

    expected_co_map = (
        hp.read_map(expected_map_filename, field=(0, 1, 2), dtype=np.float64) * u.K_CMB
    ).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(line_freq))

    assert_quantity_allclose(co_map, expected_co_map, rtol=1e-5)

    co_map = co.get_emission([114.271, 115.271, 116.271, 117.271] * u.GHz)
    # weight is about 1/3 as bandwidth is 3 GHz
    assert_quantity_allclose(co_map, expected_co_map * 0.3304377039951613, rtol=1e-3)

    co_map = co.get_emission([100, 120] * u.GHz)
    # weight is about 1/20, consider normalization also includes conversion to power units
    assert_quantity_allclose(co_map, expected_co_map * 0.05475254098360655, rtol=1e-4)


@pytest.mark.parametrize("model_tag", ["co2", "co3"])
def test_co_model(model_tag):
    include_high_galactic_latitude_clouds = model_tag == "co3"

    model = Sky(preset_strings=[model_tag], nside=16, output_unit="uK_CMB")

    co_map = model.get_emission(115.271 * u.GHz)

    tag = "wHGL" if include_high_galactic_latitude_clouds else "noHGL"
    remote_data = RemoteData()
    expected_map_filename = remote_data.get(
        f"co/testing/CO10_TQUmaps_{tag}_nside16_ring.fits.zip"
    )
    expected_co_map = (
        hp.read_map(expected_map_filename, field=(0, 1, 2), dtype=np.float64) * u.K_CMB
    )

    assert_quantity_allclose(co_map, expected_co_map, rtol=1e-5)
