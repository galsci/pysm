from . import mpi


class MapDistribution:
    def __init__(
        self, pixel_indices=None, mpi_comm=None, nside=None, smoothing_lmax=None
    ):
        """Define how a map is distributed

        In a serial environment, this is only useful if you want to generate a partial sky,
        pass an array `pixel_indices` with the indices in RING ordering.

        in a MPI environment, pass a `mpi4py` communicator and the desired :math:`\\ell_{max}`
        for smoothing and this class will create a ring-based distribution suitable for smoothing
        with `libsharp`.

        Parameters
        ----------
        pixel_indices : ndarray of integers
            Subset of pixels that should be used in RING ordering
        mpi_comm: object
            MPI communicator object (optional, default=None).
        nside: int
            Resolution parameter at which this model is to be calculated.
        smoothing_lmax : int
            :math:`\\ell_{max}` for the smoothing step, by default :math:`3*N_{side}-1`
        """
        self.pixel_indices = pixel_indices
        self.mpi_comm = mpi_comm
        self.smoothing_lmax = smoothing_lmax
        self.nside = nside

        if self.mpi_comm is not None and pixel_indices is None:
            assert (
                self.nside is not None
            ), "libsharp needs to know the NSIDE to create the distribution"
            if self.smoothing_lmax is None:
                self.smoothing_lmax = 3 * self.nside - 1

            if mpi.libsharp is None:
                self.pixel_indices = mpi.distribute_pixels_uniformly(
                    self.mpi_comm, self.nside
                )
            else:
                self.pixel_indices, lg, lo = mpi.distribute_rings_libsharp(
                    self.mpi_comm, self.nside, lmax=self.smoothing_lmax
                )
                self.libsharp_grid = lg
                self.libsharp_order = lo
