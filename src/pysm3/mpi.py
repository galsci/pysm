import healpy as hp
import numpy as np

from . import units as u
from . import utils

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
    return np.arange(
        mpi_comm.rank * pix_per_proc, min((mpi_comm.rank + 1) * pix_per_proc, npix)
    )


def expand_pix(startpix, ringpix, local_npix):
    """Turn first pixel index and number of pixel in full array of pixels
    to be optimized with cython
    """
    i = 0
    local_pix = np.zeros(local_npix)
    for start, num in zip(startpix, ringpix):
        local_pix[i : i + num] = np.arange(start, start + num)
        i += num
    return local_pix


def distribute_rings_libsharp(mpi_comm, nside, lmax):
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
    import libsharp

    nrings = 4 * nside - 1  # four missing pixels

    # ring indices are 1-based
    ring_indices_emisphere = np.arange(2 * nside, dtype=np.int32) + 1

    local_ring_indices = ring_indices_emisphere[mpi_comm.rank :: mpi_comm.size]

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
    startpix, ringpix, _, _, _ = hp.ringinfo(nside, local_ring_indices.astype(int))

    local_npix = libsharp_grid.local_size()
    local_pixels = expand_pix(startpix, ringpix, local_npix).astype(int)

    local_m_indices = np.arange(mpi_comm.rank, lmax + 1, mpi_comm.size, dtype=np.int32)
    libsharp_order = libsharp.packed_real_order(lmax, ms=local_m_indices)
    return local_pixels, libsharp_grid, libsharp_order


def assemble_map_on_rank0(comm, local_map, pixel_indices, n_components, npix):
    from mpi4py import MPI

    full_maps_rank0 = (
        np.zeros((n_components, npix), dtype=local_map.dtype)
        if comm.rank == 0
        else None
    )
    local_map_buffer = np.zeros((n_components, npix), dtype=local_map.dtype)
    local_map_buffer[:, pixel_indices] = local_map
    comm.Reduce(local_map_buffer, full_maps_rank0, root=0, op=MPI.SUM)
    return full_maps_rank0


def mpi_smoothing(input_map, fwhm, map_dist):
    import libsharp

    beam = hp.gauss_beam(
        fwhm=fwhm.to(u.rad).value, lmax=map_dist.smoothing_lmax, pol=True
    )

    input_map_I = input_map if input_map.ndim == 1 else input_map[0]
    input_map_I_contig = np.ascontiguousarray(input_map_I.reshape((1, 1, -1))).astype(
        np.float64, copy=False
    )

    alm_sharp_I = libsharp.analysis(
        map_dist.libsharp_grid,
        map_dist.libsharp_order,
        input_map_I_contig,
        spin=0,
        comm=map_dist.mpi_comm,
    )
    map_dist.libsharp_order.almxfl(alm_sharp_I, np.ascontiguousarray(beam[:, 0:1]))
    out = libsharp.synthesis(
        map_dist.libsharp_grid,
        map_dist.libsharp_order,
        alm_sharp_I,
        spin=0,
        comm=map_dist.mpi_comm,
    )[0]
    assert np.isnan(out).sum() == 0

    if utils.has_polarization(input_map):
        alm_sharp_P = libsharp.analysis(
            map_dist.libsharp_grid,
            map_dist.libsharp_order,
            np.ascontiguousarray(input_map[1:3, :].reshape((1, 2, -1))).astype(
                np.float64, copy=False
            ),
            spin=2,
            comm=map_dist.mpi_comm,
        )

        map_dist.libsharp_order.almxfl(
            alm_sharp_P, np.ascontiguousarray(beam[:, (1, 2)])
        )

        signal_map_P = libsharp.synthesis(
            map_dist.libsharp_grid,
            map_dist.libsharp_order,
            alm_sharp_P,
            spin=2,
            comm=map_dist.mpi_comm,
        )[0]
        out = np.vstack((out, signal_map_P))
    return out
