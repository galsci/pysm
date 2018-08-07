""" This submodule contains the various component models used in PySM.

Classes:
    ModifiedBlackBody
    SynchrotronPowerlaw
"""
import numpy as np
import healpy as hp
from astropy.modeling.blackbody import blackbody_nu
import astropy.units as units
from pathlib import Path
from .template import Model

class ModifiedBlackBody(Model):
    """ This is a model for modified black body emission.

    Attributes
    ----------
    I_ref, Q_ref, U_ref: ndarray
        Arrays containing the intensity or polarization reference
        templates at frequency `freq_ref_I` or `freq_ref_P`.
    """
    def __init__(self, map_I=None, map_Q=None, map_U=None, freq_ref_I=None,
                 freq_ref_P=None, map_mbb_index=None, map_mbb_temperature=None,
                 nside=None, mpi_comm=None):
        """ This function initializes the modified black body model.

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
        map_mbb_index: `pathlib.Path` object
            Path to the map to be used as the power law index for the dust
            opacity in a modified blackbody model.
        map_mbb_temperature: `pathlib.Path` object
            Path to the map to be used as the temperature of the dust in a
            modified blackbody model.
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
        self.mbb_index = read_map(map_mbb_index, nside)[None, :]
        self.mbb_temperature = read_map(map_mbb_temperature, nside)[None, :] * units.K
        self.nside = nside
        return

    def get_emission(self, freqs=None):
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
        if freqs is not None:
            if isinstance(freqs, np.ndarray):
                freqs = freqs * units.GHz
            elif isinstance(freqs, list):
                freqs = np.array(freqs) * units.GHz
            else:
                try:
                    freqs = np.array([freqs]) * units.GHz
                except:
                    print("""Could not make freqs into an ndarray, check
                    input.""")
                    raise

        outputs = []
        for freq in freqs:
            I_scal = (freq / self.freq_ref_I) ** (self.mbb_index - 2.)
            P_scal = (freq / self.freq_ref_P) ** (self.mbb_index - 2.)
            I_scal *= blackbody_ratio(freq, self.freq_ref_I,
                                      self.mbb_temperature)
            P_scal *= blackbody_ratio(freq, self.freq_ref_P,
                                      self.mbb_temperature)
            iqu_freq = np.concatenate((I_scal * self.I_ref,
                                       P_scal * self.Q_ref,
                                       P_scal * self.U_ref))
            outputs.append(iqu_freq)
        return np.array(outputs)


class SynchrotronPowerLaw(Model):
    """ This is a model for a simple power law synchrotron model.
    """
    def __init__(self, mpi_comm=None):
        """ This function initialzes the power law model of synchrotron
        emission.

        The initialization consists of reading in emission tempaltes from
        file.
        """
        Model.__init__(self, mpi_comm)
        return


def read_map(path, nside):
    """ Wrapper of `healpy.read_map` for PySM data.

    Parameters
    ----------
    path: object `pathlib.Path`, or str
        Path of HEALPix map to be read.
    nsidE: int
        Resolution at which to return map. Map is read in at whatever resolution
        it is stored, and `healpy.ud_grade` is applied.

    Returns
    -------
    ndarray
        Numpy array containing HEALPix map in RING ordering.
    """
    # read map. Add `str()` operator in case dealing with `Path` object.
    inmap = hp.read_map(str(path), field=0, verbose=False)
    return hp.ud_grade(inmap, nside_out=nside)


def blackbody_ratio(freq_to, freq_from, temp):
    """ Function to calculate the flux ratio between two frequencies for a
    blackbody at a given temperature.
    """
    return blackbody_nu(freq_to, temp) / blackbody_nu(freq_from, temp)
