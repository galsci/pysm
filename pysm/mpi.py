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

def expand_pix(startpix, ringpix, local_npix):
    """Turn first pixel index and number of pixel in full array of pixels
    to be optimized with cython
    """
    i = 0
    local_pix = np.zeros(local_npix)
    for start, num in zip(startpix, ringpix):
        local_pix[i:i + num] = np.arange(start, start + num)
        i += num
    return local_pix


def distribute_rings_libsharp(mpi_comm, nside):
    """Create a libsharp map distribution based on rings

    Build a libsharp grid object to distribute a HEALPix map
    balancing North and South distribution of rings to achieve
    the best performance on Harmonic Transforms
    Returns the grid object and the pixel indices array in RING ordering

    Parameters
    ---------
    mpi_comm : mpi4py.MPI.Comm
        mpi4py communicator
    nside : int
        nside of the map

    Returns
    -------
    grid : libsharp.healpix_grid
        libsharp object that includes metadata about HEALPix distributed rings
    local_pix : np.ndarray
        integer array of local pixel indices in the current MPI process in RING
        ordering
    """
    nrings = 4 * nside - 1  # four missing pixels

    # ring indices are 1-based
    ring_indices_emisphere = np.arange(2 * nside, dtype=np.int32) + 1

    local_ring_indices = ring_indices_emisphere[mpi_comm.rank::mpi_comm.size]

    # to improve performance, symmetric rings north/south need to be in the same rank
    # therefore we use symmetry to create the full ring indexing

    if local_ring_indices[-1] == 2 * nside:
        # has equator ring
        local_ring_indices = np.concatenate(
            [local_ring_indices[:-1], nrings - local_ring_indices[::-1] + 1]
        )
    else:
        # does not have equator ring
        local_ring_indices = np.concatenate(
            [local_ring_indices, nrings - local_ring_indices[::-1] + 1]
        )

    libsharp_grid = libsharp.healpix_grid(nside, rings=local_ring_indices)

    # returns start index of the ring and number of pixels
    startpix, ringpix, _, _, _ = hp.ringinfo(
        nside, local_ring_indices.astype(np.int64))

    local_npix = libsharp_grid.local_size()
    local_pixels = expand_pix(startpix, ringpix, local_npix)
    return local_pixels, libsharp_grid
