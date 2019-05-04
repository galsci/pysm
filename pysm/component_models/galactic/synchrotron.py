import numpy as np
from ... import units as u
from numba import njit

from ..template import Model, check_freq_input


class SynchrotronPowerLaw(Model):
    """ This is a model for a simple power law synchrotron model.
    """

    def __init__(
        self,
        map_I,
        map_Q,
        map_U,
        freq_ref_I,
        freq_ref_P,
        map_pl_index,
        nside,
        has_polarization=True,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        pixel_indices=None,
        mpi_comm=None,
    ):
        """ This function initialzes the power law model of synchrotron
        emission.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        unit_* : string or Unit
            Unit string or Unit object for all input FITS maps, if None, the input file
            should have a unit defined in the FITS header.
        freq_ref_I, freq_ref_P: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined.  They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        map_pl_index: `pathlib.Path` object
            Path to the map to be used as the power law index.
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        super().__init__(nside, pixel_indices=pixel_indices, mpi_comm=mpi_comm)
        # do model setup
        self.I_ref = self.read_map(map_I, unit=unit_I)
        # This does unit conversion in place so we do not copy the data
        # we do not keep the original unit because otherwise we would need
        # to make a copy of the array when we run the model
        self.I_ref <<= u.uK_RJ
        self.freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
        self.has_polarization = has_polarization
        if has_polarization:
            self.Q_ref = self.read_map(map_Q, unit=unit_Q)
            self.Q_ref <<= u.uK_RJ
            self.U_ref = self.read_map(map_U, unit=unit_U)
            self.U_ref <<= u.uK_RJ
            self.freq_ref_P = u.Quantity(freq_ref_P).to(u.GHz)
        self.pl_index = self.read_map(map_pl_index, unit="")
        return

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
            get_emission_numba(
                freqs.value,
                self.I_ref.value,
                self.Q_ref.value,
                self.U_ref.value,
                self.freq_ref_I.value,
                self.freq_ref_P.value,
                self.pl_index.value,
            )
            << u.uK_RJ
        )
        return outputs


@njit(parallel=True)
def get_emission_numba(freqs, I_ref, Q_ref, U_ref, freq_ref_I, freq_ref_P, pl_index):
    outputs = np.empty((len(freqs), 3, len(I_ref)), dtype=I_ref.dtype)
    I, Q, U = 0, 1, 2
    for i_freq, freq in enumerate(freqs):
        outputs[i_freq, I, :] = I_ref
        outputs[i_freq, Q, :] = Q_ref
        outputs[i_freq, U, :] = U_ref
        outputs[i_freq, I] *= (freq / freq_ref_I) ** pl_index
        outputs[i_freq, Q:] *= (freq / freq_ref_P) ** pl_index
    return outputs
