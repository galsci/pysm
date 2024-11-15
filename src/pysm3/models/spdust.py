import numpy as np
from numba import njit

from .. import units as u
from .. import utils
from .template import Model


class SpDust(Model):
    """Implementation of the SpDust2 code of (Ali-Haimoud et al 2012)
    evaluated for a Cold Neutral Medium.
    See:
    * https://arxiv.org/abs/1003.4732
    * https://arxiv.org/abs/0812.2904
    """

    def __init__(
        self,
        map_I,
        freq_ref_I,
        emissivity,
        freq_peak,
        freq_ref_peak,
        nside,
        max_nside=None,
        unit_I=None,
        map_dist=None,
    ):
        """This function initializes the spinning dust model

        Parameters
        ----------
        map_I : `pathlib.Path` object
            Paths to the map to be used as I templates.
        unit_I : string or Unit
            Unit string or Unit object for all input FITS maps, if None, the input file
            should have a unit defined in the FITS header.
        freq_ref_I : Quantity or string
            Reference frequencies at which the templates are defined.
            They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        freq_peak : `pathlib.Path` object or string
            Path to the map to be used as frequency of the peak of the emission or
            its scalar value as a Quantity or a string convertible to a Quantity
        freq_ref_peak : Quantity or string
            Reference frequency for the peak frequency map
            They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        # do model setup
        self.I_ref = self.read_map(map_I, unit=unit_I)
        # This does unit conversion in place so we do not copy the data
        # we do not keep the original unit because otherwise we would need
        # to make a copy of the array when we run the model
        self.I_ref <<= u.uK_RJ
        self.freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
        try:  # input is a number
            self.freq_peak = u.Quantity(freq_peak).to(u.GHz)
        except TypeError:  # input is a path
            self.freq_peak = self.read_map(freq_peak, unit=u.GHz)
        freq_ref_peak = u.Quantity(freq_ref_peak).to(u.GHz)
        self.freq_peak /= freq_ref_peak
        self.emissivity = self.read_txt(emissivity, unpack=True)

    @u.quantity_input
    def get_emission(self, freqs: u.Quantity[u.GHz], weights=None):
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        return (
            compute_spdust_emission_numba(
                freqs,
                weights,
                self.I_ref.value,
                self.freq_ref_I.value,
                self.freq_peak.value,
                self.emissivity,
            )
            << u.uK_RJ
        )


@njit
def compute_spdust_scaling_numba(freq, freq_ref_I, freq_peak, emissivity):
    scaled_freq = freq / freq_peak
    scaled_ref_freq = freq_ref_I / freq_peak
    return (
        (freq_ref_I / freq) ** 2
        * np.interp(scaled_freq, emissivity[0], emissivity[1])
        / np.interp(scaled_ref_freq, emissivity[0], emissivity[1])
    )


@njit(parallel=True)
def compute_spdust_emission_numba(
    freqs, weights, I_ref, freq_ref_I, freq_peak, emissivity
):
    output = np.zeros((3, len(I_ref)), dtype=np.float64)
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
        scaling = compute_spdust_scaling_numba(
            freq,
            freq_ref_I,
            freq_peak,
            emissivity.astype(np.float64),
        )
        utils.trapz_step_inplace(freqs, weights, i, scaling * I_ref, output[0])
    return output


class SpDustPol(SpDust):
    """SpDust2 model with Polarized emission"""

    def __init__(
        self,
        map_I,
        freq_ref_I,
        emissivity,
        freq_peak,
        freq_ref_peak,
        pol_frac,
        angle_Q,
        angle_U,
        nside,
        max_nside=None,
        unit_I=None,
        map_dist=None,
    ):
        super().__init__(
            map_I=map_I,
            freq_ref_I=freq_ref_I,
            emissivity=emissivity,
            freq_peak=freq_peak,
            freq_ref_peak=freq_ref_peak,
            nside=nside,
            max_nside=max_nside,
            unit_I=unit_I,
            map_dist=map_dist,
        )
        self.pol_angle = np.arctan2(self.read_map(angle_U), self.read_map(angle_Q))
        self.pol_frac = pol_frac

    @u.quantity_input
    def get_emission(self, freqs: u.Quantity[u.GHz], weights=None):
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        return (
            compute_spdust_emission_pol_numba(
                freqs,
                weights,
                self.I_ref.value,
                self.freq_ref_I.value,
                self.freq_peak.value,
                self.emissivity,
                self.pol_angle.value,
                self.pol_frac,
            )
            << u.uK_RJ
        )


@njit(parallel=True)
def compute_spdust_emission_pol_numba(
    freqs, weights, I_ref, freq_ref_I, freq_peak, emissivity, pol_angle, pol_frac
):
    output = np.zeros((3, len(I_ref)), dtype=I_ref.dtype)
    I, Q, U = 0, 1, 2
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
        scaling = compute_spdust_scaling_numba(
            freq,
            freq_ref_I,
            freq_peak,
            emissivity.astype(np.float64),
        )
        utils.trapz_step_inplace(freqs, weights, i, scaling * I_ref, output[I])
        utils.trapz_step_inplace(
            freqs, weights, i, scaling * I_ref * pol_frac * np.cos(pol_angle), output[Q]
        )
        utils.trapz_step_inplace(
            freqs, weights, i, scaling * I_ref * pol_frac * np.sin(pol_angle), output[U]
        )
    return output
