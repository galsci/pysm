import numpy as np
from .. import units as u
from numba import njit

from .template import Model, check_freq_input


class SpDust(Model):
    """ Implementation of the SpDust2 code of (Ali-Haimoud et al 2012)
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
        unit_I=None,
        pixel_indices=None,
        mpi_comm=None,
    ):
        """ This function initializes the spinning dust model

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
        super().__init__(nside=nside, pixel_indices=pixel_indices, mpi_comm=mpi_comm)
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
    def get_emission(self, freqs: u.GHz):
        """ This function evaluates the component model at a either
        a single frequency, an array of frequencies, or over a bandpass.

        Parameters
        ----------
        freqs: float
            Frequency at which the model should be evaluated, assumed to be
            given in GHz.

        Returns
        -------
        ndarray
            Set of maps at the given frequency or frequencies. This will have
            shape (nfreq, 3, npix).
        """
        freqs = check_freq_input(freqs)
        outputs = (
            compute_spdust_scaling_numba(
                freqs.value,
                self.I_ref.value,
                self.freq_ref_I.value,
                self.freq_peak.value,
                self.emissivity,
            )
            << u.uK_RJ
        )
        return outputs


@njit(parallel=True)
def compute_spdust_scaling_numba(freqs, I_ref, freq_ref_I, freq_peak, emissivity):
    outputs = np.empty((len(freqs), 1, len(I_ref)), dtype=I_ref.dtype)
    for i_freq, freq in enumerate(freqs):
        scaled_freq = freq / freq_peak
        scaled_ref_freq = freq_ref_I / freq_peak
        outputs[i_freq, 0] = (
            I_ref
            * (freq_ref_I / freq) ** 2
            * np.interp(scaled_freq, emissivity[0], emissivity[1])
            / np.interp(scaled_ref_freq, emissivity[0], emissivity[1])
        )
    return outputs


class SpDustPol(SpDust):
    """SpDust2 model with Polarized emission"""

    def __init__(
        self,
        map_I,
        freq_ref_I,
        emissivity,
        freq_peak,
        freq_ref_peak,
        nside,
        unit_I=None,
        pol_frac=None,
        angle_Q=None,
        angle_U=None,
        pixel_indices=None,
        mpi_comm=None,
    ):
        super().__init__(
            map_I,
            freq_ref_I,
            emissivity,
            freq_peak,
            freq_ref_peak,
            nside,
            unit_I,
            pixel_indices,
            mpi_comm,
        )
