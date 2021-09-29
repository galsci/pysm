from ..utils import RemoteData

import numpy as np
import healpy as hp

import pytest

from .. import COLines


@pytest.mark.parametrize("include_high_galactic_latitude_clouds", [False, True])
def test_co(include_high_galactic_latitude_clouds):

    co = COLines(
        target_nside=16,
        output_units="K_CMB",
        has_polarization=True,
        line="10",
        include_high_galactic_latitude_clouds=include_high_galactic_latitude_clouds,
        polarization_fraction=0.001,
        theta_high_galactic_latitude_deg=20.0,
        random_seed=1234567,
        verbose=False,
        run_mcmole3d=False,
    )

    co_map = co.signal()

    tag = "wHGL" if include_high_galactic_latitude_clouds else "noHGL"
    remote_data = RemoteData()
    expected_map_filename = remote_data.get(
        "co/testing/CO10_TQUmaps_{}_nside16_ring.fits.zip".format(tag)
    )
    expected_co_map = hp.read_map(expected_map_filename, field=(0, 1, 2))

    np.testing.assert_allclose(co_map, expected_co_map, rtol=1e-5)
