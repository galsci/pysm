import healpy as hp
import numpy as np
import pytest

from pysm3 import InterpolatingComponent
from pysm3 import units as u

nside = 64
shape = (3, hp.nside2npix(nside))


@pytest.fixture
def create_maps(tmp_path):
    hp.write_map(tmp_path / "10.fits", np.ones(shape, dtype=np.float32))
    hp.write_map(tmp_path / "20.fits", 2 * np.ones(shape, dtype=np.float32))
    return tmp_path


@pytest.fixture
def interp(create_maps):
    """Setup the interpolating component"""

    return InterpolatingComponent(
        create_maps, "uK_RJ", nside, interpolation_kind="linear"
    )


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


@pytest.fixture
def interp_pre_smoothed(tmp_path):
    """Setup the interpolating component"""
    m = np.zeros(shape, dtype=np.float32)
    m[0] += 10

    m[0, hp.ang2pix(nside, np.pi / 2, 0)] = 100
    hp.write_map(tmp_path / "10.fits", m)

    return m, InterpolatingComponent(
        tmp_path,
        "uK_RJ",
        nside,
        interpolation_kind="linear",
        available_nside=[nside],
        pre_applied_beam={str(nside): 5},
        pre_applied_beam_units="deg",
    )


def test_presmoothed_null(interp_pre_smoothed):
    input_map, interp_pre_smoothed = interp_pre_smoothed
    output_map = interp_pre_smoothed.get_emission(
        10 * u.GHz,
        fwhm=5 * u.deg,
        lmax=1.5 * nside,
    )
    np.testing.assert_allclose(input_map, output_map.value)


def test_presmoothed(tmp_path):
    """Setup the interpolating component"""
    m = np.ones(shape, dtype=np.float32) * 10
    m[0, hp.ang2pix(nside, np.pi / 2, 0)] = 100
    m_smoothed = hp.smoothing(m, fwhm=np.radians(3))
    hp.write_map(tmp_path / "10.fits", m_smoothed)

    c = InterpolatingComponent(
        tmp_path,
        "uK_RJ",
        nside,
        interpolation_kind="linear",
        available_nside=[nside],
        pre_applied_beam={str(nside): 3},
        pre_applied_beam_units="deg",
    )

    input_map = hp.smoothing(m, fwhm=np.radians(5))
    output_map = c.get_emission(10 * u.GHz, fwhm=5 * u.deg, lmax=1.5 * nside)
    np.testing.assert_allclose(input_map, output_map.value, rtol=1e-3, atol=1e-4)
