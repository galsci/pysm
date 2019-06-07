import pysm
import numpy as np
import healpy as hp
import pytest

try:
    from mpi4py import MPI
except ImportError:
    pytest.skip("mpi4py failed to import, skip MPI tests", allow_module_level=True)

def test_serial_read_alm(tmp_path):
    """This is a serial test but requires libsharp and mpi4py"""
    filename = tmp_path / "alm.fits"
    mpi_comm = MPI.COMM_WORLD
    lmax = 16
    map_dist = pysm.MapDistribution(nside=64, smoothing_lmax=lmax, mpi_comm=mpi_comm)
    alm_size = hp.Alm.getsize(lmax)
    alm = np.arange(alm_size*3, dtype=np.complex64).reshape((3, alm_size))
    alm += 1j * (np.arange(alm_size*3, dtype=np.complex64).reshape((3, alm_size)))/10
    alm[:, :lmax] = alm[:, :lmax].real

    hp.write_alm(filename, alm, overwrite=True)

    local_alm = pysm.models.template.read_alm(filename, map_dist=map_dist)

    for i_pol in range(3):
        np.testing.assert_allclose(local_alm[0, i_pol, :lmax+1], i_pol*alm_size + np.arange(lmax+1))
        np.testing.assert_allclose(local_alm[0, i_pol, lmax+1::2], i_pol*alm_size + np.arange(lmax+1, alm_size))
        np.testing.assert_allclose(local_alm[0, i_pol, lmax+2::2], (i_pol*alm_size + np.arange(lmax+1, alm_size))/10)
