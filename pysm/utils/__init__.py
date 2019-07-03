# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.

import numpy as np
from numba import njit

from .. import units as u


def has_polarization(m):
    """Checks if a map or a group of map is polarized

    Works with a map of shape (IQU, npix) or
    (channels, IQU, npix)"""
    if isinstance(m, np.ndarray):  # array
        if m.ndim == 1:
            return False
        else:
            return m.shape[-2] == 3
    elif isinstance(m[0], np.ndarray):  # IQU tuple
        return len(m) == 3
    elif isinstance(m[0][0], np.ndarray):  # multiple IQU tuples
        return len(m[0]) == 3
    else:
        raise TypeError("Map format not understood")


def normalize_weights(freqs, weights):
    if freqs.isscalar or len(freqs) == 1:
        return np.array([1.0])
    else:
        if weights is None:
            weights = np.ones(len(freqs), dtype=np.float)
        weights = (weights * u.uK_RJ).to_value((u.Jy / u.sr), equivalencies=u.cmb_equivalencies(freqs))
        return weights / np.trapz(weights, freqs.value)


@njit
def trapz_step_inplace(freqs, weights, i, m, output):
    """Execute a step of the trapezoidal rule and accumulate into output

    freqs : ndarray
        Frequency axis, generally in GHz, but doesn't matter as long as
        weights were normalized accordingly
    weights : ndarray
        Frequency bandpass response, normalized to unit integral (with trapz)
    i : integer
        Index of the current step in the arrays
    m : ndarray
        Emission evaluated at the current frequency to be accumulated
    output : ndarray
        Array where the integrated emission is accumulated.
    """
    # case for a single frequency, compensate for the .5 factor
    if i == 0 and len(freqs) == 1:
        delta_freq = 2
    # first step of the integration
    elif i == 0:
        delta_freq = freqs[1] - freqs[0]
    # last step
    elif i == (len(freqs) - 1):
        delta_freq = freqs[-1] - freqs[-2]
    # middle steps
    else:
        delta_freq = freqs[i + 1] - freqs[i - 1]
    output += 0.5 * m * weights[i] * delta_freq

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
