"""Test the `apply_smoothing_and_coord_transform` function"""

import healpy as hp
import numpy as np

try:
    import pixell.enmap
    import pixell.reproject
except ImportError:
    pass

import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3.units as u
from pysm3 import apply_smoothing_and_coord_transform

INITIAL_FWHM = (1 * u.deg).to_value(u.radian)
FWHM = (5 * u.deg).to_value(u.radian)
NSIDE = 128
CAR_RESOL = 12 * u.arcmin
LMAX = int(NSIDE * 1.5)


@pytest.fixture(
    scope="module"
)  # scope makes the fixture just run once per execution of module
def input_map():
    beam_window = hp.gauss_beam(fwhm=INITIAL_FWHM, lmax=LMAX) ** 2
    cl = np.zeros((6, len(beam_window)))
    cl[0:3] = beam_window
    np.random.seed(7)
    return hp.synfast(cl, NSIDE, lmax=LMAX, new=True) * u.uK_RJ


def test_smoothing_healpix_beamwindow(input_map):
    beam_window = hp.gauss_beam(fwhm=FWHM, lmax=LMAX, pol=True).T

    smoothed_map = apply_smoothing_and_coord_transform(
        input_map, lmax=LMAX, beam_window=beam_window
    )
    assert input_map.shape[0] == 3
    assert smoothed_map.shape == input_map.shape
    assert_quantity_allclose(
        actual=smoothed_map,
        desired=hp.smoothing(input_map, fwhm=FWHM, lmax=LMAX, use_pixel_weights=True)
        * input_map.unit,
        rtol=1e-7,
    )


def test_smoothing_healpix(input_map):
    smoothed_map = apply_smoothing_and_coord_transform(
        input_map, lmax=LMAX, fwhm=FWHM * u.radian
    )
    assert input_map.shape[0] == 3
    assert smoothed_map.shape == input_map.shape
    assert_quantity_allclose(
        actual=smoothed_map,
        desired=hp.smoothing(input_map, fwhm=FWHM, lmax=LMAX, use_pixel_weights=True)
        * input_map.unit,
    )


def test_car_nosmoothing(input_map):
    # `enmap_from_healpix` has no iteration or weights
    # so for test purpose we reproduce it here
    alm = (
        hp.map2alm(input_map, lmax=LMAX, iter=0, use_pixel_weights=False)
        * input_map.unit
    )
    car_map = apply_smoothing_and_coord_transform(
        alm,
        input_alm=True,
        fwhm=None,
        return_healpix=False,
        return_car=True,
        output_car_resol=CAR_RESOL,
        lmax=LMAX,
    )
    assert car_map.shape == (3, 900, 1800)
    shape, wcs = pixell.enmap.fullsky_geometry(
        CAR_RESOL.to_value(u.radian), dims=(3,), variant="fejer1"
    )
    map_rep = (
        pixell.reproject.enmap_from_healpix(
            input_map, shape, wcs, lmax=LMAX, rot=None, ncomp=3
        )
    )
    assert_quantity_allclose(actual=car_map, desired=map_rep)


def test_healpix_output_nside(input_map):
    output_nside = 64
    output_map = apply_smoothing_and_coord_transform(
        input_map, fwhm=None, output_nside=output_nside, lmax=LMAX
    )
    assert output_map.shape == (3, hp.nside2npix(output_nside))
    alm = hp.map2alm(input_map, use_pixel_weights=True, lmax=LMAX)
    desired = (
        hp.alm2map(
            alm,
            nside=output_nside,
        )
        * input_map.unit
    )
    assert_quantity_allclose(
        actual=output_map,
        desired=desired,
        rtol=1e-7,
    )
