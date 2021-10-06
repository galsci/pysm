from pathlib import Path
import healpy as hp

import numpy as np
from numba import njit
from astropy import constants as const

from .. import units as u
from .. import utils
from .template import Model
from .dust import blackbody_ratio


class ModifiedBlackBodyLayers(Model):
    """This is a model for modified black body emission.

    Attributes
    ----------
    I_ref, Q_ref, U_ref: ndarray
        Arrays containing the intensity or polarization reference
        templates at frequency `freq_ref_I` or `freq_ref_P`.
    """

    def __init__(
        self,
        map_layers,
        freq_ref,
        map_mbb_index,
        map_mbb_temperature,
        nside,
        num_layers=1,
        unit_layers=None,
        unit_mbb_temperature=None,
        map_dist=None,
    ):
        """This function initializes the modified black body model.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        map_layers: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        unit_* : string or Unit
            Unit string or Unit object for all input FITS maps, if None, the input file
            should have a unit defined in the FITS header.
        freq_ref_I, freq_ref_P: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        map_mbb_index: `pathlib.Path` object or scalar value
            Path to the map to be used as the power law index for the dust
            opacity in a modified blackbody model, for a constant value use
            a float or an integer
        map_mbb_temperature: `pathlib.Path` object or scalar
            Path to the map to be used as the temperature of the dust in a
            modified blackbody model. For a constant value use a float or an
            integer
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        super().__init__(nside=nside, map_dist=map_dist)
        num_pix = hp.nside2npix(nside)
        self.num_layers = num_layers
        self.layers = np.empty((num_layers, 3, num_pix))
        if isinstance(map_layers, (str, Path)):
            for i_layer in range(num_layers):
                self.layers[i_layer, 3, :] = self.read_map(
                    map_layers.format(layer=i_layer), field=(0, 1, 2), unit=unit
                )
        else:
            self.layers = u.Quantity(map_layers, unit_layers)

        self.freq_ref = u.Quantity(freq_ref).to(u.GHz)

        with u.set_enabled_equivalencies(u.cmb_equivalencies(self.freq_ref)):
            self.layers <<= u.uK_RJ

        self.mbb_temperature = np.empty((num_layers, num_pix))

        if isinstance(map_mbb_index, (str, Path)):
            self.mbb_index = np.empty((num_layers, num_pix))
            for i_layer in range(num_layers):
                self.mbb_index[i_layer] = self.read_map(
                    map_mbb_index.format(layer=i_layer), unit=""
                )
        else:
            self.mbb_index = u.Quantity(map_mbb_index, unit="")

        if isinstance(map_mbb_temperature, (str, Path)):
            self.mbb_temperature = np.empty((num_layers, num_pix))
            for i_layer in range(num_layers):
                self.mbb_temperature[i_layer] = self.read_map(
                    map_mbb_temperature.format(layer=i_layer), unit=unit_mbb_temperature
                )
        else:
            self.mbb_temperature = u.Quantity(
                map_mbb_temperature, unit=unit_mbb_temperature
            )

        self.mbb_temperature <<= u.K
        self.nside = int(nside)

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        outputs = get_emission_numba(
            freqs,
            weights,
            self.layers.value,
            self.freq_ref.value,
            self.mbb_index.value,
            self.mbb_temperature.value,
        )
        return outputs << u.uK_RJ


@njit(parallel=True)
def get_emission_numba(
    freqs,
    weights,
    layers,
    freq_ref,
    mbb_index,
    mbb_temperature,
):
    npix = layers.shape[-1]
    output = np.zeros((3, npix), dtype=layers.dtype)
    temp = np.zeros((3, npix), dtype=layers.dtype)

    I, Q, U = 0, 1, 2
    for i, (freq, weight) in enumerate(zip(freqs, weights)):
        for i_layer in range(layers.shape[0]):
            temp[:, :] = layers[i_layer, :, :]
            # -2 because black body is in flux unit and not K_RJ
            temp[I] *= (freq / freq_ref) ** (mbb_index[i_layer] - 2.0)
            temp[I] *= blackbody_ratio(freq, freq_ref, mbb_temperature[i_layer])
            freq_scaling_P = (freq / freq_ref) ** (
                mbb_index[i_layer] - 2.0
            ) * blackbody_ratio(freq, freq_ref, mbb_temperature[i_layer])
            for P in [Q, U]:
                temp[P] *= freq_scaling_P
            if len(freqs) > 1:
                utils.trapz_step_inplace(freqs, weights, i, temp, output)
            else:
                output += temp

    return output
