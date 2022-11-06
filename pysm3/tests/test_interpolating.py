import numpy as np
import healpy as hp

from .. import InterpolatingComponent
from .. import units as u


def test_interpolating(tmp_path):

    nside = 4
    shape = (3, hp.nside2npix(nside))
    hp.write_map(tmp_path / "10.fits", np.ones(shape, dtype=np.float32))
    hp.write_map(tmp_path / "20.fits", 2 * np.ones(shape, dtype=np.float32))

    interp = InterpolatingComponent(
        tmp_path, "uK_RJ", nside, interpolation_kind="linear"
    )

    interpolated_map = interp.get_emission(15 * u.GHz)
    np.testing.assert_allclose(1.5 * np.ones(shape) * u.uK_RJ, interpolated_map)

    interpolated_map = interp.get_emission(19 * u.GHz)
    np.testing.assert_allclose(1.9 * np.ones(shape) * u.uK_RJ, interpolated_map)

    # test pick one of the available maps
    interpolated_map = interp.get_emission(20 * u.GHz)
    np.testing.assert_allclose(2 * np.ones(shape) * u.uK_RJ, interpolated_map)

def test_interpolating_bandpass_boundary(tmp_path):

    nside = 4
    shape = (3, hp.nside2npix(nside))
    hp.write_map(tmp_path / "10.fits", np.ones(shape, dtype=np.float32))
    hp.write_map(tmp_path / "20.fits", 2 * np.ones(shape, dtype=np.float32))

    interp = InterpolatingComponent(
        tmp_path, "uK_RJ", nside, interpolation_kind="linear"
    )

    interpolated_map = interp.get_emission(np.array([15, 20])*u.GHz)
    np.testing.assert_allclose(1.5 * np.ones(shape) * u.uK_RJ, interpolated_map)
