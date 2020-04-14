""" This submodule contains the tempalte for the `Model` object.
The available PySM models are subclasses of this template, and
when adding models to PySM it is advised that the user subclasses
this template, ensuring that the new subclass has the required
`get_emission` method.

Objects:
    Model
"""
import warnings
import numpy as np
import healpy as hp
from astropy.io import fits
from .. import utils
from .. import units as u
from .. import mpi
import gc


class Model:
    """ This is the template object for PySM objects.

    If a MPI communicator is passed as input and `pixel_indices` is None,
    the class automatically distributes the maps across processes.
    You can implement your own pixel distribution passing both a MPI
    communicator and `pixel_indices`, however that won't support smoothing
    with `libsharp`.
    If `libsharp` is available, the rings are distributed as expected
    by `libsharp` to perform distributed spherical harmonics transforms,
    see :py:func:`pysm.mpi.distribute_rings_libsharp`, the `libsharp` grid
    object is saved in `self.libsharp_grid`.
    If libsharp is not available, pixels are distributed uniformly across
    processes, see :py:func:`pysm.mpi.distribute_pixels_uniformly`"""

    def __init__(self, nside, map_dist=None):
        """
        Parameters
        ----------
        mpi_comm: object
            MPI communicator object (optional, default=None).
        nside: int
            Resolution parameter at which this model is to be calculated.
        smoothing_lmax : int
            :math:`\ell_{max}` for the smoothing step, by default :math:`2*N_{side}`
        """
        self.nside = nside
        assert nside is not None
        self.map_dist = map_dist

    def read_map(self, path, unit=None, field=0, nside=None):
        """Wrapper of the PySM read_map function that automatically
        uses nside, pixel_indices and mpi_comm defined in this Model
        by default.
        If the `nside` keyword is set, this will override the `Model`
        value when reading the map. This can be used to read in data
        products that must be processed at a specific nside.
        """
        if nside is not None:
            nside = nside
        else:
            nside = self.nside
        return read_map(path, nside, unit=unit, field=field, map_dist=self.map_dist)

    def read_txt(self, path, **kwargs):
        mpi_comm = None if self.map_dist is None else self.map_dist.mpi_comm
        return read_txt(path, mpi_comm=mpi_comm, **kwargs)

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        """ This function evaluates the component model at a either
        a single frequency, an array of frequencies, or over a bandpass.

        Parameters
        ----------
        freqs: scalar or array astropy.units.Quantity
            Frequency at which the model should be evaluated, in a frequency
            which can be converted to GHz using astropy.units.
            If an array of frequencies is provided, integrate using trapz
            with a equal weighting, i.e. simulate a top-hat bandpass.
        weights: np.array, optional
            Array of weights describing the frequency response of the instrument,
            i.e. the bandpass. Weights are normalized and applied in Jy/sr.

        Returns
        -------
        output : astropy.units.Quantity
            Simulated map at the given frequency or integrated over the given
            bandpass. The shape of the output is (3,npix) for polarized components,
            (1,npix) for temperature-only components. Output is in `uK_RJ`.
        """
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        outputs = np.zeros((3, hp.nside2npix(self.nside)), dtype=np.float32)
        return outputs << u.uK_RJ


def apply_smoothing_and_coord_transform(
    input_map, fwhm=None, rot=None, lmax=None, map_dist=None
):
    """Apply smoothing and coordinate rotation to an input map

    it applies the `healpy.smoothing` Gaussian smoothing kernel if `map_dist`
    is None, otherwise applies distributed smoothing with `libsharp`.
    In the distributed case, no rotation is supported.

    Parameters
    ----------
    input_map : ndarray
        Input map, of shape `(3, npix)`
        This is assumed to have no beam at this point, as the
        simulated small scale tempatle on which the simulations are based
        have no beam.
    fwhm : astropy.units.Quantity
        Full width at half-maximum, defining the
        Gaussian kernels to be applied.
    rot: hp.Rotator
        Apply a coordinate rotation give a healpy `Rotator`, e.g. if the
        inputs are in Galactic, `hp.Rotator(coord=("G", "C"))` rotates
        to Equatorial

    Returns
    -------
    smoothed_map : np.ndarray
        Array containing the smoothed sky
    """

    if map_dist is None:
        nside = hp.get_nside(input_map)
        alm = hp.map2alm(
            input_map, lmax=lmax, use_pixel_weights=True if nside > 16 else False
        )
        if fwhm is not None:
            hp.smoothalm(
                alm, fwhm=fwhm.to_value(u.rad), verbose=False, inplace=True, pol=True
            )
        if rot is not None:
            rot.rotate_alm(alm, inplace=True)
        smoothed_map = hp.alm2map(alm, nside=nside, verbose=False, pixwin=False)

    else:
        assert (rot is None) or (
            rot.coordin == rot.coordout
        ), "No rotation supported in distributed smoothing"
        smoothed_map = mpi.mpi_smoothing(input_map, fwhm, map_dist)

    if hasattr(input_map, "unit"):
        smoothed_map <<= input_map.unit
    return smoothed_map


def apply_normalization(freqs, weights):
    """ Function to apply a normalization constraing to a set of weights.
    This imposes the requirement that the integral of the weights over the
    array `freqs` must equal unity.

    Parameters
    ----------
    freqs: ndarray
        Array containing the domain over which to integrate.
    weights: ndarray
        Array containing the samples to integrate.

    Returns
    -------
    tuple(ndarray)
        Tuple containing the frequencies and weights. These are numpy arrays
        of equal length.
    """
    return freqs, weights / np.trapz(weights, freqs)


def extract_hdu_unit(path):
    """ Function to extract unit from an hdu.
    Parameters
    ----------
    path: Path object
        Path to the fits file.
    Returns
    -------
    string
        String specifying the unit of the fits data.
    """
    hdul = fits.open(path)
    try:
        unit = hdul[1].header["TUNIT1"]
    except KeyError:
        # in the case that TUNIT1 does not exist, assume unitless quantity.
        unit = ""
        warnings.warn("No physical unit associated with file " + str(path))
    return unit


def read_map(path, nside, unit=None, field=0, map_dist=None):
    """Wrapper of `healpy.read_map` for PySM data. This function also extracts
    the units from the fits HDU and applies them to the data array to form an
    `astropy.units.Quantity` object.
    This function requires that the fits file contains a TUNIT key for each
    populated field.

    Parameters
    ----------
    path : object `pathlib.Path`, or str
        Path of HEALPix map to be read.
    nside : int
        Resolution at which to return map. Map is read in at whatever resolution
        it is stored, and `healpy.ud_grade` is applied.

    Returns
    -------
    map : ndarray
        Numpy array containing HEALPix map in RING ordering.
    """
    mpi_comm = None if map_dist is None else map_dist.mpi_comm
    pixel_indices = None if map_dist is None else map_dist.pixel_indices
    filename = utils.RemoteData().get(path)

    if (mpi_comm is not None and mpi_comm.rank == 0) or (mpi_comm is None):
        output_map = hp.read_map(filename, field=field, verbose=False, dtype=None)
        dtype = output_map.dtype
        # numba only supports little endian
        if dtype.byteorder == ">":
            dtype = dtype.newbyteorder()
        # mpi4py has issues if the dtype is a string like ">f4"
        if dtype == np.dtype(np.float32):
            dtype = np.dtype(np.float32)
        elif dtype == np.dtype(np.float64):
            dtype = np.dtype(np.float64)
        nside_in = hp.get_nside(output_map)
        if nside < nside_in:  # do downgrading in double precision
            output_map = hp.ud_grade(output_map.astype(np.float64), nside_out=nside)
        else:
            output_map = hp.ud_grade(output_map, nside_out=nside)
        output_map = output_map.astype(dtype, copy=False)
        if unit is None:
            unit = extract_hdu_unit(filename)
        shape = output_map.shape
    elif mpi_comm is not None and mpi_comm.rank > 0:
        npix = hp.nside2npix(nside)
        try:
            ncomp = len(field)
        except TypeError:  # field is int
            ncomp = 1
        shape = npix if ncomp == 1 else (len(field), npix)
        unit = ""
        dtype = None

    if mpi_comm is not None:
        from mpi4py import MPI

        dtype = mpi_comm.bcast(dtype, root=0)
        unit = mpi_comm.bcast(unit, root=0)

        node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
        mpi_type = MPI._typedict[dtype.char]
        mpi_type_size = mpi_type.Get_size()
        win = MPI.Win.Allocate_shared(
            np.prod(shape) * mpi_type_size if node_comm.rank == 0 else 0,
            mpi_type_size,
            comm=node_comm,
        )
        shared_buffer, item_size = win.Shared_query(0)
        assert item_size == mpi_type_size
        shared_buffer = np.array(shared_buffer, dtype="B", copy=False)
        node_shared_map = np.ndarray(buffer=shared_buffer, dtype=dtype, shape=shape)

        # only the first MPI process in each node is in this communicator
        rank_comm = mpi_comm.Split(0 if node_comm.rank == 0 else MPI.UNDEFINED)
        if mpi_comm.rank == 0:
            node_shared_map[:] = output_map
        if node_comm.rank == 0:
            rank_comm.Bcast(node_shared_map, root=0)

        mpi_comm.barrier()
        # code with broadcast to whole communicator
        # if mpi_comm.rank > 0:
        #     output_map = np.empty(shape, dtype=dtype)
        # mpi_comm.Bcast(output_map, root=0)
    else:  # without MPI node_shared_map is just another reference to output_map
        node_shared_map = output_map

    if pixel_indices is not None:
        # make copies so that Python can release the full array
        try:  # multiple components
            output_map = np.array(
                [each[pixel_indices].copy() for each in node_shared_map]
            )
        except IndexError:  # single component
            output_map = node_shared_map[pixel_indices].copy()

    if mpi_comm is not None:
        del node_shared_map
        del shared_buffer
        win.Free()
        gc.collect()

    return u.Quantity(output_map, unit, copy=False)


def read_txt(path, mpi_comm=None, **kwargs):
    """MPI-aware numpy.loadtxt function
    reads text file on rank 0 with np.loadtxt and broadcasts over MPI

    Parameters
    ----------
    path : str
        path to fits file.
    mpi_comm :  mpi4py MPI Communicator.

    Returns
    -------
    output : numpy.ndarray
        data read with numpy.loadtxt
    """

    filename = utils.RemoteData().get(path)

    if (mpi_comm is not None and mpi_comm.rank == 0) or (mpi_comm is None):
        output = np.loadtxt(filename, **kwargs)
    elif mpi_comm is not None and mpi_comm.rank > 0:
        output = None

    if mpi_comm is not None:
        output = mpi_comm.bcast(output, root=0)

    return output
