import numpy as np
import healpy as hp

from .. import InterpolatingComponent
from .. import units as u

import pytest

nside = 4
shape = (3, hp.nside2npix(nside))


@pytest.fixture
def interp(tmp_path):
    """Setup the interpolating component"""
    hp.write_map(tmp_path / "10.fits", np.ones(shape, dtype=np.float32))
    hp.write_map(tmp_path / "20.fits", 2 * np.ones(shape, dtype=np.float32))

    interp = InterpolatingComponent(
        tmp_path, "uK_RJ", nside, interpolation_kind="linear"
    )
    return interp


def test_interpolating(interp):

    interpolated_map = interp.get_emission(15 * u.GHz)
    np.testing.assert_allclose(1.5 * np.ones(shape) * u.uK_RJ, interpolated_map)

    interpolated_map = interp.get_emission(19 * u.GHz)
    np.testing.assert_allclose(1.9 * np.ones(shape) * u.uK_RJ, interpolated_map)

    # test pick one of the available maps
    interpolated_map = interp.get_emission(20 * u.GHz)
    np.testing.assert_allclose(2 * np.ones(shape) * u.uK_RJ, interpolated_map)


def test_interpolating_bandpass_boundary_above(interp):

    interpolated_map = interp.get_emission(np.array([15, 20]) * u.GHz)
    np.testing.assert_allclose(
        1.82 * np.ones(shape) * u.uK_RJ, interpolated_map, rtol=1e-2
    )


def test_interpolating_bandpass_boundary_below(interp):
    interpolated_map = interp.get_emission([10, 12] * u.GHz)
    np.testing.assert_allclose(
        1.118 * np.ones(shape) * u.uK_RJ, interpolated_map, rtol=1e-2
    )
