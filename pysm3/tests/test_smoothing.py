"""Test the `apply_smoothing_and_coord_transform` function"""

import healpy as hp
import numpy as np

try:
    import pixell.enmap
    import pixell.reproject
except ImportError:
    pass

from pysm3.models import apply_smoothing_and_coord_transform
import pysm3.units as u

from astropy.tests.helper import assert_quantity_allclose
import pytest

"""
    input_map,
    fwhm=None,
    rot=None,
    lmax=None,
    output_nside=None,
    output_car_resol=None,
    return_healpix=True,
    return_car=False,
    map_dist=None,
"""


FWHM = (5 * u.deg).to_value(u.radian)
NSIDE = 128
CAR_RESOL = 12 * u.arcmin
LMAX = int(NSIDE * 1.5)

# scope makes the fixture just run once per execution of this module
@pytest.fixture(scope="module")
def input_map():
    beam_window = hp.gauss_beam(fwhm=FWHM, lmax=LMAX) ** 2
    cl = np.zeros((6, len(beam_window)))
    cl[0:3] = beam_window
    m = hp.synfast(cl, NSIDE, lmax=LMAX, new=True) * u.uK_RJ
    return m


def test_smoothing_healpix(input_map):

    smoothed_map = apply_smoothing_and_coord_transform(input_map, fwhm=FWHM * u.radian)
    assert smoothed_map.shape == input_map.shape
    assert_quantity_allclose(
        actual=smoothed_map,
        desired=hp.smoothing(input_map, fwhm=FWHM, lmax=LMAX, use_pixel_weights=True),
    )


def test_car_nosmoothing(input_map):

    smoothed_map = apply_smoothing_and_coord_transform(
        input_map,
        fwhm=None,
        return_healpix=False,
        return_car=True,
        output_car_resol=CAR_RESOL,
        lmax=LMAX,
    )
    assert smoothed_map.shape == (901, 1800)
    shape, wcs = pixell.enmap.fullsky_geometry(CAR_RESOL.to_value(u.radian))
    map_rep = pixell.reproject.enmap_from_healpix(
        input_map, shape, wcs, lmax=LMAX, rot=None
    )[0]
    assert_quantity_allclose(actual=smoothed_map, desired=map_rep)
