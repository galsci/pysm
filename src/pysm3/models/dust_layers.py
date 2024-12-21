from pathlib import Path

import healpy as hp
import numpy as np
from numba import njit

from .. import units as u
from .. import utils
from .dust import blackbody_ratio
from .template import Model


class ModifiedBlackBodyLayers(Model):
    def __init__(
        self,
        map_layers,
        freq_ref,
        map_mbb_index,
        map_mbb_temperature,
        nside,
        color_correction=1,
        max_nside=None,
        num_layers=1,
        unit_layers=None,
        unit_mbb_temperature=None,
        map_dist=None,
    ):
        """Modified Black Body model with multiple layers

        Used for the MKD 3D dust model by
        Ginés Martínez-Solaeche, Ata Karakci, Jacques Delabrouille:
        https://arxiv.org/abs/1706.04162

        Parameters
        ----------
        map_layers: `pathlib.Path`, str or ndarray
            Path or string with a templated layer number {layer} to download or access locally
            a IQU map for each layer (1-based layer number)
            Alternatively an array of shape (num_layers, 3, num_pix)
        num_layers: int
            Number of layers
        unit_* : str or Unit
            Unit string or Unit object for all input FITS maps, if None, the input file
            should have a unit defined in the FITS header.
        freq_ref: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        map_mbb_index, map_mbb_temperature: `pathlib.Path`, str or ndarray
            Path or string with a templated 1-based layer number {layer} with
            the spectra index or the blackbody temperature.
            Alternatively an array of shape (num_layers, num_pix)
        nside: int
            Resolution parameter at which this model is to be calculated (with `ud_grade`)
        color_correction: float
            Scalar correction factor multiplied to the maps, implemented to add
            a color correction factor to Planck HFI 353 GHz maps
        """
        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        num_pix = hp.nside2npix(nside)
        self.num_layers = num_layers
        self.layers = u.Quantity(np.empty((num_layers, 3, num_pix)), unit=unit_layers)
        if isinstance(map_layers, (str, Path)):
            for i_layer in range(num_layers):
                self.layers[i_layer, :, :] = self.read_map(
                    map_layers.format(layer=i_layer + 1),
                    field=(0, 1, 2),
                    unit=unit_layers,
                )
        else:
            self.layers = u.Quantity(map_layers, unit_layers)

        self.color_correction = color_correction
        self.layers *= self.color_correction

        self.freq_ref = u.Quantity(freq_ref).to(u.GHz)

        with u.set_enabled_equivalencies(u.cmb_equivalencies(self.freq_ref)):
            self.layers <<= u.uK_RJ

        if isinstance(map_mbb_index, (str, Path)):
            self.mbb_index = u.Quantity(np.empty((num_layers, num_pix)), unit="")
            for i_layer in range(num_layers):
                self.mbb_index[i_layer] = self.read_map(
                    map_mbb_index.format(layer=i_layer + 1), unit=""
                )
        else:
            self.mbb_index = u.Quantity(map_mbb_index, unit="")

        if isinstance(map_mbb_temperature, (str, Path)):
            self.mbb_temperature = u.Quantity(
                np.empty((num_layers, num_pix)), unit_mbb_temperature
            )
            for i_layer in range(num_layers):
                self.mbb_temperature[i_layer] = self.read_map(
                    map_mbb_temperature.format(layer=i_layer + 1),
                    unit=unit_mbb_temperature,
                )
        else:
            self.mbb_temperature = u.Quantity(
                map_mbb_temperature, unit=unit_mbb_temperature
            )

        self.mbb_temperature <<= u.K
        self.nside = int(nside)

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
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
    output = np.zeros((3, npix), dtype=np.float64)
    temp = np.zeros((3, npix), dtype=np.float64)

    I, Q, U = 0, 1, 2
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
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
