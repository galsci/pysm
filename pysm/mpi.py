import healpy as hp
import numpy as np

try:
    import libsharp
except ImportError:
    libsharp = None

def distribute_pixels_uniformly(mpi_comm, nside):
    """Define the pixel_indices for each MPI rank to distribute a map uniformly

    Parameters
    ----------
    mpi_comm : mpi4py.MPI.Comm
        mpi4py communicator
    nside : int
        nside of the map

    Returns
    -------
    pixel_indices : np.ndarray
        local pixel indices
    """
    npix = hp.nside2npix(nside)
    pix_per_proc = int(np.ceil(npix / mpi_comm.size))
    pixel_indices = np.arange(
        mpi_comm.rank * pix_per_proc,
        min((mpi_comm.rank + 1) * pix_per_proc, npix),
    )
    return pixel_indices

