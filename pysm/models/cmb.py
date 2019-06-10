import numpy as np

from .. import utils
from .. import units as u
from .template import Model, check_freq_input

class CMBMap(Model):

    def __init__(self, map_I, map_Q, map_U, nside, map_dist=None):
        super().__init__(nside=nside, map_dist=map_dist)
        self.map_I = self.read_map(map_I, unit=u.uK_CMB)
        self.map_Q = self.read_map(map_Q, unit=u.uK_CMB)
        self.map_U = self.read_map(map_U, unit=u.uK_CMB)

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        freqs = check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        convert_to_uK_RJ = (np.ones(len(freqs), dtype=np.double) * u.uK_CMB).to_value(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs)
        )

        if len(freqs) == 1:
            scaling_factor = convert_to_uK_RJ[0]
        else:
            scaling_factor = np.trapz(convert_to_uK_RJ * weights, x=freqs.value)

        print("scaling_factor", scaling_factor)

        return (np.array([self.map_I, self.map_Q, self.map_U]) * scaling_factor) << u.uK_RJ
