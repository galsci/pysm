import numpy as np
import healpy as hp

from .. import InterpolatingComponent
from .. import units as u


def test_interpolating(tmp_path):

    nside = 4
    shape = (3, hp.nside2npix(nside))
    hp.write_map(tmp_path / "10.fits", np.zeros(shape, dtype=np.float32))
    hp.write_map(tmp_path / "20.fits", np.ones(shape, dtype=np.float32))

    interp = InterpolatingComponent(
        tmp_path, "uK_RJ", nside, interpolation_kind="linear", has_polarization=True
    )

    interpolated_map = interp.get_emission(15 * u.GHz)
    np.testing.assert_allclose(0.5 * np.ones(shape) * u.uK_RJ, interpolated_map)

    # test pick one of the available maps
    interpolated_map = interp.get_emission(20 * u.GHz)
    np.testing.assert_allclose(np.ones(shape) * u.uK_RJ, interpolated_map)
