import os.path
from pathlib import Path

from numba import njit
import numpy as np

import pysm3.units as u
from .interpolating import InterpolatingComponent
from .. import utils

@njit
def y2uK_CMB(nu):
    """Compton-y distortion at a given frequency
    Parmeters:
    nu (float): frequency in GHz
    Returns:
    float: intensity variation dT_CMB in micro-Kelvin
      dT_CMB = dI_nu / (dB_nu / dT)_Tcmb
      where B_nu is the Planck function and dI_nu is the intensity distortion
    """

    h = 6.62607004e-27
    k = 1.380622e-16
    Tcmb = 2.725
    x = h * nu * 1e9 / k / Tcmb
    return 1e6 * Tcmb * (x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4)


class WebSkyCIB(InterpolatingComponent):
    """PySM component interpolating between precomputed maps"""

    def __init__(
        self,
        local_folder=None,
        websky_version="0.3",
        input_units="MJy / sr",
        nside=4096,
        max_nside=8192,
        interpolation_kind="linear",
        map_dist=None,
        verbose=False
    ):
        self.local_folder = local_folder
        super().__init__(
            path=websky_version,
            input_units=input_units,
            nside=nside,
            max_nside=max_nside,
            interpolation_kind=interpolation_kind,
            map_dist=map_dist,
            verbose=verbose,
        )
        self.remote_data = utils.RemoteData()

    def get_filenames(self, path):
        """Get filenames for a websky version
        For a standard interpolating component, we list files in folder,
        here we need to know the names in advance so that we can only download the required maps
        """

        websky_version = path
        if websky_version in "0.3":

            available_frequencies = [27, 39, 93, 100, 145, 217, 225, 280, 353, 545, 857]

            filenames = {
                freq: "websky/0.3/cib_{:04d}.fits".format(freq)
                                    # freq, nside="512/" if self.nside <= 512 else "")
                for freq in available_frequencies
            }
        if self.local_folder is not None:
            for freq in filenames:
                filenames[freq] = os.path.join(self.local_folder, filenames[freq])

        return filenames

    def read_map_by_frequency(self, freq):
        filename = self.remote_data.get(self.maps[freq])
        return self.read_map_file(freq, filename)


