import os
import healpy as hp
from numba import njit, types
from numba.typed import Dict
import numpy as np
from .template import Model
from .. import units as u
from .. import utils
from ..utils import trapz_step_inplace
import warnings

import logging

log = logging.getLogger("pysm3")


class InterpolatingComponent(Model):
    def __init__(
        self,
        path,
        input_units,
        nside,
        max_nside=None,
        interpolation_kind="linear",
        map_dist=None,
        verbose=False,
    ):
        """PySM component interpolating between precomputed maps

        In order to save memory, maps are converted to float32, if this is not acceptable, please
        open an issue on the PySM repository.
        When you create the model, PySM checks the folder of the templates and stores a list of
        available frequencies. Once you call `get_emission`, maps are read, ud_graded to the target
        nside and stored for future use. This is useful if you are running many channels
        with a similar bandpass.
        If not, you can call `cached_maps.clear()` to remove the cached maps.

        It always returns a IQU map to avoid broadcasting issues, even
        if the inputs are I only.

        Parameters
        ----------
        path : str
            Path should contain maps named as the frequency in GHz
            e.g. 20.fits or 20.5.fits or 00100.fits
        input_units : str
            Any unit available in PySM3 e.g. "uK_RJ", "uK_CMB"
        nside : int
            HEALPix NSIDE of the output maps
        interpolation_kind : string
            Currently only linear is implemented
        map_dist : pysm.MapDistribution
            Required for partial sky or MPI, see the PySM docs
        verbose : bool
            Control amount of output
        """

        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        self.maps = {}
        self.maps = self.get_filenames(path)

        # use a numba typed Dict so we can used in JIT compiled code
        self.cached_maps = Dict.empty(
            key_type=types.float64, value_type=types.float32[:, :]
        )

        self.freqs = np.array(list(self.maps.keys()))
        self.freqs.sort()
        self.input_units = input_units
        self.interpolation_kind = interpolation_kind
        self.verbose = verbose

    def get_filenames(self, path):
        # Override this to implement name convention
        filenames = {}
        for f in os.listdir(path):
            if f.endswith(".fits"):
                freq = float(os.path.splitext(f)[0])
                filenames[freq] = os.path.join(path, f)
        return filenames

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        nu = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(nu, weights)

        if len(nu) == 1:

            # special case: we request only 1 frequency and that is among the ones
            # available as input
            check_isclose = np.isclose(self.freqs, nu[0])
            if np.any(check_isclose):

                freq = self.freqs[check_isclose][0]
                out = self.read_map_by_frequency(freq)
                if out.ndim == 1 or out.shape[0] == 1:
                    zeros = np.zeros_like(out)
                    out = np.array([out, zeros, zeros])
                return out << u.uK_RJ

        npix = hp.nside2npix(self.nside)
        if nu[0] < self.freqs[0]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz < lower bound {} GHz".format(
                    nu[0], self.freqs[0]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ
        if nu[-1] > self.freqs[-1]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz > upper bound {} GHz".format(
                    nu[-1], self.freqs[-1]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ

        freq_range = utils.get_relevant_frequencies(self.freqs, nu[0], nu[-1])
        log.info("Frequencies considered: %s", str(freq_range))

        for freq in freq_range:
            if freq not in self.cached_maps:
                m = self.read_map_by_frequency(freq)
                if m.shape[0] != 3:
                    m = m.reshape((1, -1))
                self.cached_maps[freq] = m.astype(np.float32)
                for i_pol, pol in enumerate("IQU" if m.shape[0] == 3 else "I"):
                    log.info(
                        "Mean emission at {} GHz in {}: {:.4g} uK_RJ".format(
                            freq, pol, self.cached_maps[freq][i_pol].mean()
                        )
                    )

        out = compute_interpolated_emission_numba(
            nu, weights, freq_range, self.cached_maps
        )

        if out.ndim == 1 or out.shape[0] == 1:
            if out.ndim == 2:
                out = out[0]
            zeros = np.zeros_like(out)
            out = np.array([out, zeros, zeros])
        # the output of out is always 2D, (IQU, npix)
        return out << u.uK_RJ

    def read_map_by_frequency(self, freq):
        filename = self.maps[freq]
        return self.read_map_file(freq, filename)

    def read_map_file(self, freq, filename):
        log.info("Reading map %s", filename)

        try:
            m = self.read_map(
                filename,
                field=(0, 1, 2),
                unit=self.input_units,
            )
        except IndexError:
            m = self.read_map(
                filename,
                field=0,
                unit=self.input_units,
            )
        return m.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz)).value


@njit(parallel=False)
def compute_interpolated_emission_numba(freqs, weights, freq_range, all_maps):
    output = np.zeros(
        all_maps[freq_range[0]].shape, dtype=all_maps[freq_range[0]].dtype
    )
    if len(freqs) > 1:
        temp = np.zeros_like(output)
    else:
        temp = output
    index_range = np.arange(len(freq_range))
    for i in range(len(freqs)):
        interpolation_weight = np.interp(freqs[i], freq_range, index_range)
        int_interpolation_weight = int(interpolation_weight)
        relative_weight = interpolation_weight - int_interpolation_weight
        temp[:] = (1 - relative_weight) * all_maps[freq_range[int_interpolation_weight]]
        if relative_weight > 0:
            temp += relative_weight * all_maps[freq_range[int_interpolation_weight + 1]]

        if len(freqs) > 1:
            trapz_step_inplace(freqs, weights, i, temp, output)
    return output
