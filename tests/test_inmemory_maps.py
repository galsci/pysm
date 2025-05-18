import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
import pysm3
import pysm3.units as u
import pysm3.models as models


def test_dust_inmemory():
    nside = 32
    npix = 12 * nside ** 2
    arr = np.ones((3, npix)) * 10 * u.uK_RJ
    model = models.ModifiedBlackBody(
        map_I=arr,
        freq_ref_I=353 * u.GHz,
        freq_ref_P=353 * u.GHz,
        map_mbb_index=np.ones(npix) * u.dimensionless_unscaled,
        map_mbb_temperature=np.ones(npix) * 20 * u.K,
        nside=nside,
    )
    freq = 353 * u.GHz
    simulated_map = model.get_emission(freq)
    expected = arr
    assert_quantity_allclose(simulated_map, expected)


def test_synchrotron_inmemory():
    nside = 32
    npix = 12 * nside ** 2
    arr = np.ones((3, npix)) * 5 * u.uK_RJ
    model = models.PowerLaw(
        map_I=arr,
        freq_ref_I=30 * u.GHz,
        freq_ref_P=30 * u.GHz,
        map_pl_index=np.ones(npix) * -3.0 * u.dimensionless_unscaled,
        nside=nside,
    )
    freq = 30 * u.GHz
    simulated_map = model.get_emission(freq)
    expected = arr
    assert_quantity_allclose(simulated_map, expected)


def test_dust_inmemory_IQU_args():
    nside = 32
    npix = 12 * nside ** 2
    arr_I = np.ones(npix) * 7 * u.uK_RJ
    arr_Q = np.ones(npix) * 2 * u.uK_RJ
    arr_U = np.ones(npix) * 3 * u.uK_RJ
    arr = np.stack([arr_I, arr_Q, arr_U])
    model = models.ModifiedBlackBody(
        map_I=arr,
        freq_ref_I=353 * u.GHz,
        freq_ref_P=353 * u.GHz,
        map_mbb_index=np.ones(npix) * u.dimensionless_unscaled,
        map_mbb_temperature=np.ones(npix) * 20 * u.K,
        nside=nside,
    )
    freq = 353 * u.GHz
    simulated_map = model.get_emission(freq)
    expected = arr
    assert_quantity_allclose(simulated_map, expected)


def test_synchrotron_inmemory_I_arg():
    nside = 32
    npix = 12 * nside ** 2
    arr_I = np.ones(npix) * 11 * u.uK_RJ
    model = models.PowerLaw(
        map_I=arr_I,
        freq_ref_I=30 * u.GHz,
        map_pl_index=np.ones(npix) * -3.0 * u.dimensionless_unscaled,
        nside=nside,
        has_polarization=False,
    )
    freq = 30 * u.GHz
    simulated_map = model.get_emission(freq)
    expected = np.zeros((3, npix)) * u.uK_RJ
    expected[0] = arr_I
    assert_quantity_allclose(simulated_map, expected)
