import pytest
import numpy as np
import healpy as hp
import pysm
import astropy.units as u

try:
    from mpi4py import MPI
except ImportError:
    pytest.skip("mpi4py failed to import, skip MPI tests", allow_module_level=True)

try:
    import libsharp
except ImportError:
    pytest.skip("libsharp failed to import, skip MPI smoothing tests", allow_module_level=True)


@pytest.fixture
def mpi_comm():
    comm = MPI.COMM_WORLD
    return comm


def test_mpi_smoothing(mpi_comm):
    nside = 128
    lmax = 2 * nside
    model = pysm.Model(
        nside, mpi_comm=mpi_comm, pixel_indices=None, smoothing_lmax=lmax
    )
    distributed_map = model.read_map("pysm_2/dust_temp.fits")
    fwhm = 5 * u.deg
    smoothed_distributed_map = model.mpi_smoothing(distributed_map, fwhm)
    full_map_rank0 = pysm.mpi.assemble_map_on_rank0(
        mpi_comm,
        smoothed_distributed_map,
        model.pixel_indices,
        n_components=1,
        npix=hp.nside2npix(nside),
    )[0]
    if mpi_comm.rank == 0:
        np.testing.assert_allclose(
            full_map_rank0,
            hp.smoothing(
                pysm.read_map("pysm_2/dust_temp.fits", nside=nside).value,
                fwhm.to(u.rad).value,
                iter=0,
                lmax=lmax,
                use_pixel_weights=False,
            ),
            rtol=1e-5,
        )
