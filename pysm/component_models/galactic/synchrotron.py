import numpy as np
import healpy as hp
from ... import units
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
        self.__pl_index = read_map(map_pl_index, nside) * units.dimensionless_unscaled

        freq_ref_I = float(freq_ref_I) * units.GHz
        freq_ref_P = float(freq_ref_P) * units.GHz
        self.__iqu_ref_freqs = units.Quantity([freq_ref_I] + 2 * [freq_ref_P])

        npix = hp.nside2npix(nside)
        self.__iqu_ref = np.empty((3, npix)) * units.uK_RJ
        self.__iqu_ref[0] = read_map(map_I, nside) * units.uK_RJ
        self.__iqu_ref[1] = read_map(map_Q, nside) * units.uK_RJ
        self.__iqu_ref[2] = read_map(map_U, nside) * units.uK_RJ
        return

    @units.quantity_input(freqs=units.GHz, equivalencies=units.spectral())
    def get_emission(self, freqs) -> units.uK_RJ:
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
        # ensure freqs in units of GHz, and reshape scalars to one dimensional array.
        freqs = check_freq_input(freqs)
        # calculate scaled templates, shape is (nfreqs, npol, npix)
        return pl_sed(freqs[:, None, None], self.__iqu_ref_freqs[None, :, None],
                      self.__pl_index[None, None, :]) * self.__iqu_ref[None, ...]

@units.quantity_input(freqs_to=units.GHz, freqs_from=units.GHz, index=units.dimensionless_unscaled)
def pl_sed(freqs_to, freqs_from, index) -> units.dimensionless_unscaled:
    """ Power law SED.

    FIXME: need to decide on how to make units of scaling explicit. i.e. whether index is beta or beta-2

    Parameters
    ----------
    freqs_to: Quantity
    freqs_from: Quantity
    index: Quantity

    Returns
    -------
    Quantity
    """
    freqs_to = freqs_to.to(units.GHz, equivalencies=units.spectral())
    freqs_from = freqs_from.to(units.GHz, equivalencies=units.spectral())
    return (freqs_to / freqs_from) ** index
