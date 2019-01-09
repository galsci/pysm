import numpy as np
import astropy.units as units
from ..template import Model, check_freq_input, read_map

class SynchrotronPowerLaw(Model):
    """ This is a model for a simple power law synchrotron model.
    """
    def __init__(self, map_I=None, map_Q=None, map_U=None, freq_ref_I=None,
                 freq_ref_P=None, map_pl_index=None, nside=None, mpi_comm=None):
        """ This function initialzes the power law model of synchrotron
        emission.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        freq_ref_I, freq_ref_P: float
            Reference frequencies at which the intensity and polarization
            templates are defined.
        map_pl_index: `pathlib.Path` object
            Path to the map to be used as the power law index.
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        Model.__init__(self, mpi_comm)
        # do model setup
        self.I_ref = read_map(map_I, nside)[None, :] * units.uK
        self.Q_ref = read_map(map_Q, nside)[None, :] * units.uK
        self.U_ref = read_map(map_U, nside)[None, :] * units.uK
        self.freq_ref_I = freq_ref_I * units.GHz
        self.freq_ref_P = freq_ref_P * units.GHz
        self.pl_index = read_map(map_pl_index, nside)[None, :]
        self.nside = nside
        return

    def get_emission(self, freqs):
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
        # freqs must be given in GHz.
        freqs = check_freq_input(freqs)
        outputs = []
        for freq in freqs:
            I_scal = (freq / self.freq_ref_I) ** self.pl_index
            P_scal = (freq / self.freq_ref_P) ** self.pl_index
            iqu_freq = np.concatenate((I_scal * self.I_ref,
                                       P_scal * self.Q_ref,
                                       P_scal * self.U_ref))
            outputs.append(iqu_freq)
        return np.array(outputs)
