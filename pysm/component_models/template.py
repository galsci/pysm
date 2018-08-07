""" This submodule contains the tempalte for the `Model` object.
The available PySM models are subclasses of this template, and
when adding models to PySM it is advised that the user subclasses
this template, ensuring that the new subclass has the required
`get_emission` method.

Objects:
    Model
"""
import numpy as np
import healpy as hp
import astropy.units as units

class Model(object):
    """ This is the template object for PySM objects.
    """
    def __init__(self, mpi_comm=None):
        """
        Parameters:
        ----------
        mpi_comm: object
            MPI communicator object (optional, default=None).
        """
        self.mpi_comm = mpi_comm
        return

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
            fwhms = np.array(fwhms) * units.arcmin
        elif isinstance(fwhms, np.ndarray):
            fwhms *= units.arcmin
        else:
            fwhms = np.array([fwhms]) * units.arcmin
            try:
                assert(fwhms.ndim < 2)
            except AssertionError:
                print("""Check that FWHMs is given as a 1D list, 1D array.
                of float""")
        out = []
        for sky, fwhm in zip(skies, fwhms):
            out.append(hp.smoothing(sky, fwhm=fwhm.to(units.rad) / units.rad,
                       verbose=False))
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
