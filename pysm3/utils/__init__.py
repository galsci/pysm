# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.

import numpy as np
from numba import njit

from .. import units as u
from .data import RemoteData  # noqa: F401
from .logpoltens import log_pol_tens_to_map, map_to_log_pol_tens  # noqa: F401
from .small_scales import sigmoid  # noqa: F401
from .add_metadata import add_metadata  # noqa: F401

import logging

log = logging.getLogger("pysm3")


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
    """Normalize bandpass weights to support integration "in K_RJ"

    Bandpasses are assumed to be in power units, i.e. Jy/sr
    this function takes the input weights and multiplies them
    by the conversion factor from RJ to Jy/sr, so that when
    we do the integration with foregrounds defined in RJ units
    using these weights, we first convert to power and do the
    integration in power.
    Then they are also all multiplied by the integrated conversion
    factor from Jy/sr to RJ, so that the output of the integral
    is transformed back to RJ.

    Parameters
    ----------
    freqs : np.array
        array of frequency in GHz, without units
    weights : np.array
        array of weights, without units, if None, a top-hat
        bandpass (in power units) will be used

    Returns
    -------
    normalized_weights : np.array
        Normalized weights
    """
    if len(freqs) == 1:
        return np.array([1.0])
    else:
        if weights is None:
            weights = np.ones(len(freqs), dtype=np.float)
        weights = weights / np.trapz(weights, freqs)
        weights = (weights * u.uK_RJ).to_value(
            (u.Jy / u.sr), equivalencies=u.cmb_equivalencies(freqs * u.GHz)
        )
        return weights / np.trapz(weights, freqs)


def bandpass_unit_conversion(
    freqs, weights=None, output_unit=None, input_unit=u.uK_RJ, cut=1e-10
):
    """Unit conversion from input to output unit given a bandpass

    The bandpass is always assumed in power units (Jy/sr)
    Gain weights below cut are removed.

    Parameters
    ----------
    freqs : astropy.units.Quantity
        Frequency array in a unit compatible with GHz
    weights : numpy array
        Bandpass weights, if None, assume top-hat bandpass
        Weights are always assumed in (Jy/sr), whatever the
        input unit
    output_unit : astropy.units.Unit
        Output unit for the bandpass conversion factor
    input_unit : astropy.units.Unit
        Input unit for the bandpass conversion factor
        Default uK_RJ, the standard unit used internally by PySM
    cut : float
        Normalized gains under this value are removed

    Returns
    -------
    factor : astropy.units.Quantity
        Conversion factor in units of output_unit/input_unit
    """
    assert output_unit is not None, "Please specify an output unit"
    freqs = check_freq_input(freqs)
    if len(freqs) == 1:
        factor = (1.0 * input_unit).to_value(
            output_unit, equivalencies=u.cmb_equivalencies(freqs * u.GHz)
        )
    else:
        if weights is None:
            weights = np.ones(len(freqs), dtype=np.float64)
        else:
            weights = weights.copy()
        weights /= np.trapz(weights, freqs)
        if weights.min() < cut:
            good = np.logical_not(weights < cut)
            log.info(
                "Removing {}/{} points below {}".format(good.sum(), len(good), cut)
            )
            weights = weights[good]
            freqs = freqs[good]
            weights /= np.trapz(weights, freqs)
        weights_to_rj = (weights * input_unit).to_value(
            (u.Jy / u.sr), equivalencies=u.cmb_equivalencies(freqs * u.GHz)
        )
        weights_to_out = (weights * output_unit).to_value(
            (u.Jy / u.sr), equivalencies=u.cmb_equivalencies(freqs * u.GHz)
        )
        factor = np.trapz(weights_to_rj, freqs) / np.trapz(weights_to_out, freqs)
    return factor * u.Unit(output_unit / input_unit)


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
    """Function to check that the input to `Model.get_emission` is a
    np.ndarray.

    This function will convert input scalar frequencies
    to a Quantity array in GHz

    Parameters
    ----------
    freqs: astropy.units.Quantity
        Input frequency array

    Returns
    -------
    freqs : np.array
        Frequencies in GHz in a numpy array
    """
    if freqs.isscalar:
        freqs = freqs[None]
    return freqs.to_value(u.GHz)
