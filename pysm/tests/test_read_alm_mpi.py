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
    filename = str(tmp_path / "alm.fits")
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank

    lmax = 16
    map_dist = pysm.MapDistribution(nside=64, smoothing_lmax=lmax, mpi_comm=mpi_comm)

    alm_size = hp.Alm.getsize(lmax)
    if rank == 0:
        alm = np.arange(alm_size * 3, dtype=np.complex64).reshape((3, alm_size))
        alm += (
            1j
            * (np.arange(alm_size * 3, dtype=np.complex64).reshape((3, alm_size)))
            / 10
        )
        alm[:, :lmax] = alm[:, :lmax].real

        hp.write_alm(filename, alm, overwrite=True)

    mpi_comm.Barrier()

    local_alm = pysm.models.template.read_alm(filename, map_dist=map_dist)

    for i_pol in range(3):
        if rank == 0:
            # m = 0
            np.testing.assert_allclose(
                local_alm[0, i_pol, : lmax + 1], i_pol * alm_size + np.arange(lmax + 1)
            )
            if mpi_comm.size > 1:
                # m = 2
                np.testing.assert_allclose(
                    local_alm[0, i_pol, lmax + 1 : 3 * lmax - 1 : 2],
                    i_pol * alm_size
                    + np.arange(lmax + 1 + lmax, (lmax + 1) + (lmax) + (lmax - 1)),
                )
        elif rank == 1:
            # m = 1
            np.testing.assert_allclose(
                local_alm[0, i_pol, : 2 * lmax : 2],
                i_pol * alm_size + np.arange(lmax + 1, (lmax + 1) + (lmax)),
            )
