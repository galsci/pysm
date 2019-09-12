""" This submodule contains the tempalte for the `Model` object.
The available PySM models are subclasses of this template, and
when adding models to PySM it is advised that the user subclasses
this template, ensuring that the new subclass has the required
`get_emission` method.

Objects:
    Model
"""
import warnings
import os.path
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.io import fits
from astropy.utils import data
from .. import utils
from ..constants import DATAURL
from .. import mpi

from unittest.mock import Mock


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

    def __init__(self, nside, map_dist=None, dataurl=None):
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
        self.dataurl = dataurl

    def read_map(self, path, unit=None, field=0):
        """Wrapper of the PySM read_map function that automatically
        uses nside, pixel_indices and mpi_comm defined in this Model
        """
        return read_map(
            path,
            self.nside,
            unit=unit,
            field=field,
            map_dist=self.map_dist,
            dataurl=self.dataurl,
        )

    def read_txt(self, path, **kwargs):
        mpi_comm = None if self.map_dist is None else self.map_dist.mpi_comm
        return read_txt(path, mpi_comm=mpi_comm, **kwargs)


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


def read_map(path, nside, unit=None, field=0, map_dist=None, dataurl=None):
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
    if dataurl is None:
        dataurl = DATAURL
    # read map. Add `str()` operator in case dealing with `Path` object.
    if os.path.exists(str(path)):  # Python 3.5 requires turning a Path object to str
        filename = str(path)
    else:
        with data.conf.set_temp("dataurl", dataurl), data.conf.set_temp(
            "remote_timeout", 30
        ):
            filename = data.get_pkg_data_filename(path)
    # inmap = hp.read_map(filename, field=field, verbose=False)
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
        dtype = mpi_comm.bcast(dtype, root=0)
        if mpi_comm.rank > 0:
            output_map = np.empty(shape, dtype=dtype)
        mpi_comm.Bcast(output_map, root=0)
        unit = mpi_comm.bcast(unit, root=0)

    if pixel_indices is not None:
        # make copies so that Python can release the full array
        try:  # multiple components
            output_map = np.array([each[pixel_indices].copy() for each in output_map])
        except IndexError:  # single component
            output_map = output_map[pixel_indices].copy()

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

    if os.path.exists(str(path)):
        filename = str(path)
    else:
        with data.conf.set_temp("dataurl", DATAURL), data.conf.set_temp(
            "remote_timeout", 30
        ):
            filename = data.get_pkg_data_filename(path)

    if (mpi_comm is not None and mpi_comm.rank == 0) or (mpi_comm is None):
        output = np.loadtxt(filename, **kwargs)
    elif mpi_comm is not None and mpi_comm.rank > 0:
        output = None

    if mpi_comm is not None:
        output = mpi_comm.bcast(output, root=0)

    return output
