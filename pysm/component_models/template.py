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
from ..constants import DATAURL


class Model(object):
    """ This is the template object for PySM objects."""

    def __init__(self, mpi_comm=None, nside=None, pixel_indices=None):
        """
        Parameters
        ----------
        mpi_comm: object
            MPI communicator object (optional, default=None).
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        self.nside = nside
        self.mpi_comm = mpi_comm
        self.pixel_indices = pixel_indices
        return

    def read_map(self, path, field=0):
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
        # read map. Add `str()` operator in case dealing with `Path` object.
        if os.path.exists(
            str(path)
        ):  # Python 3.5 requires turning a Path object to str
            filename = str(path)
        else:
            with data.conf.set_temp("dataurl", DATAURL), data.conf.set_temp(
                "remote_timeout", 30
            ):
                filename = data.get_pkg_data_filename(path)
        # inmap = hp.read_map(filename, field=field, verbose=False)
        if (self.mpi_comm is not None and self.mpi_comm.rank == 0) or (
            self.mpi_comm is None
        ):
            output_map = hp.ud_grade(
                hp.read_map(filename, field=field, verbose=False), nside_out=self.nside
            )
            unit_string = extract_hdu_unit(filename)
        elif self.mpi_comm is not None and self.mpi_comm.rank > 0:
            npix = hp.nside2npix(self.nside)
            try:
                ncomp = len(field)
            except TypeError:  # field is int
                ncomp = 1
            shape = npix if ncomp == 1 else (len(field), npix)
            output_map = np.empty(shape, dtype=np.float64)

        if self.mpi_comm is not None:
            self.mpi_comm.Bcast(output_map, root=0)
            unit_string = self.mpi_comm.bcast(unit_string, root=0)

        if self.pixel_indices is None:
            return output_map
        else:
            try:  # multiple components
                return u.Quantity(
                    np.array([each[self.pixel_indices] for each in output_map]),
                    unit_string,
                )
            except IndexError:  # single component
                return u.Quantity(output_map[self.pixel_indices], unit_string)

    def apply_bandpass(self, bpasses):
        """ Method to calculate the emission averaged over a bandpass.

        Note: this method may be overridden by child classes which require more
        complicated implementations of bandpass integration, as long as they are
        compatible with the input and output of this template.

        Parameters
        ----------
        bandpass: list(dict)
            List of dictionaries. Each dictionary contains 'freqs' and 'weights'
            which give the range of frequencies over which the bandpass is
            sensitive, and the correpsonding weight.

        Returns
        -------
        list(dict)
            The same list of dictionaries, updated with a 'response' keyword,
            containing the sky response to this bandpass.
        """
        out = []
        for (freqs, weights) in bpasses:
            freqs, weights = apply_normalization(freqs, weights)
            weight_emission = self.get_emission(freqs) * weights[:, None, None]
            # NOTE THIS CURRENTLY ASSUMES THAT THE BANDPASS IS GIVEN IN UNITS OF
            # UKRJ. THIS SHOULD BE MADE EXPLICIT.
            out.append(np.trapz(weight_emission, freqs, axis=0))
        return np.array(out)

    def apply_smoothing(self, skies, fwhms):
        """ Method to apply smoothing to a set of simulations. This currently
        applies only the `healpy.smoothing` Gaussian smoothing kernel, but will
        be updated with a more general functionality.

        Note: this method may be overridden by child classes which require more
        complicated implementations of smoothing, as long as they are compatible
        with the input and output of this template.

        Parameters
        ----------
        skies: ndarray
            Numpy array of shape (nchannels, 3, npix), containing the unsmoothed
            skies. This is assumed to have no beam at this point, as the
            simulated small scale tempalte on which the simulations are based
            have no beam.
        fwhms: list(float)
            List of full width at half-maixima in arcminutes, defining the
            Gaussian kernels to be applied.

        Returns
        -------
        ndarray
            Array containing the smoothed skies.
        """
        if isinstance(fwhms, list):
            fwhms = np.array(fwhms) * u.arcmin
        elif isinstance(fwhms, np.ndarray):
            fwhms *= u.arcmin
        else:
            fwhms = np.array([fwhms]) * u.arcmin
            try:
                assert fwhms.ndim < 2
            except AssertionError:
                print(
                    """Check that FWHMs is given as a 1D list, 1D array.
                of float"""
                )
        out = []
        for sky, fwhm in zip(skies, fwhms):
            out.append(hp.smoothing(sky, fwhm=fwhm.to(u.rad) / u.rad, verbose=False))
        return np.array(out)


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


def check_freq_input(freqs):
    """ Function to check that the input to `Model.get_emission` is a
    np.ndarray.

    This function will convet input integers or arrays to a single element
    numpy array.

    Parameters
    ----------
    freqs: int, float, list, ndarray

    Returns
    -------
    ndarray
        Frequencies in numpy array form.
    """
    if isinstance(freqs, np.ndarray):
        freqs = freqs
    elif isinstance(freqs, list):
        freqs = np.array(freqs)
    else:
        try:
            freqs = np.array([freqs])
        except:
            print(
                """Could not make freqs into an ndarray, check
            input."""
            )
            raise
    if isinstance(freqs, u.Quantity):
        if freqs.isscalar:
            return freqs[None]
        return freqs
    return freqs * u.GHz


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
