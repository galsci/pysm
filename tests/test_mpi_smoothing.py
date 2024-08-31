import healpy as hp
import numpy as np
import pytest

import pysm3
import pysm3.units as u

try:
    from mpi4py import MPI
except ImportError:
    pytest.skip("mpi4py failed to import, skip MPI tests", allow_module_level=True)

try:
    import libsharp  # noqa: F401
except ImportError:
    pytest.skip(
        "libsharp failed to import, skip MPI smoothing tests", allow_module_level=True
    )


@pytest.fixture
def mpi_comm():
    return MPI.COMM_WORLD


def test_mpi_assemble(mpi_comm):
    nside = 128
    map_dist = pysm3.MapDistribution(pixel_indices=None, mpi_comm=mpi_comm, nside=nside)
    model = pysm3.Model(nside, map_dist=map_dist)
    distributed_map = model.read_map("pysm_2/dust_temp.fits")
    full_map_rank0 = pysm3.mpi.assemble_map_on_rank0(
        mpi_comm,
        distributed_map,
        model.map_dist.pixel_indices,
        n_components=1,
        npix=hp.nside2npix(nside),
    )[0]
    if mpi_comm.rank == 0:
        np.testing.assert_allclose(
            full_map_rank0,
            pysm3.read_map("pysm_2/dust_temp.fits", nside=nside).value,
            rtol=1e-5,
        )


def test_mpi_smoothing(mpi_comm):
    nside = 128
    lmax = 2 * nside
    map_dist = pysm3.MapDistribution(
        pixel_indices=None, mpi_comm=mpi_comm, smoothing_lmax=lmax, nside=nside
    )
    model = pysm3.Model(nside, map_dist=map_dist)
    distributed_map = model.read_map("pysm_2/dust_temp.fits")
    fwhm = 5 * u.deg
    smoothed_distributed_map = pysm3.mpi_smoothing(
        distributed_map, fwhm, map_dist=map_dist
    )
    full_map_rank0 = pysm3.mpi.assemble_map_on_rank0(
        mpi_comm,
        smoothed_distributed_map,
        model.map_dist.pixel_indices,
        n_components=1,
        npix=hp.nside2npix(nside),
    )[0]
    if mpi_comm.rank == 0:
        np.testing.assert_allclose(
            full_map_rank0,
            hp.smoothing(
                pysm3.read_map("pysm_2/dust_temp.fits", nside=nside).value,
                fwhm.to(u.rad).value,
                iter=0,
                lmax=lmax,
                use_pixel_weights=False,
            ),
            rtol=1e-5,
        )
