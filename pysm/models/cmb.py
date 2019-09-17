import numpy as np

from .. import utils
from .. import units as u
from .template import Model


class CMBMap(Model):
    def __init__(self, map_IQU, nside, map_dist=None):
        super().__init__(nside=nside, map_dist=map_dist)
        self.map = self.read_map(map_IQU, unit=u.uK_CMB, field=(0, 1, 2))

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        convert_to_uK_RJ = (np.ones(len(freqs), dtype=np.double) * u.uK_CMB).to_value(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs)
        )

        if len(freqs) == 1:
            scaling_factor = convert_to_uK_RJ[0]
        else:
            scaling_factor = np.trapz(convert_to_uK_RJ * weights, x=freqs.value)

        return u.Quantity(self.map.value * scaling_factor, unit=u.uK_RJ, copy=False)
