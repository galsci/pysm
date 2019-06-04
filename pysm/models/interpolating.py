import os
from numba import njit
import numpy as np
from scipy.interpolate import interp1d
from .template import Model, check_freq_input
from .. import units as u
from .. import utils
from pysm.utils import trapz_step_inplace

import healpy as hp


class InterpolatingComponent(Model):
    def __init__(
        self,
        path,
        input_units,
        nside,
        interpolation_kind="linear",
        has_polarization=True,
        map_dist=None,
        verbose=False,
    ):
        """PySM component interpolating between precomputed maps

        Parameters
        ----------
        path : str
            Path should contain maps named as the frequency in GHz e.g. 20.fits or 20.5.fits or 00100.fits
        input_units : str
            Any unit available in PySM (see `pysm.convert_units` e.g. `Jysr`, `MJsr`, `uK_RJ`, `K_CMB`).
        nside : int
            HEALPix NSIDE of the output maps
        interpolation_kind : string
            Currently only linear is implemented
        has_polarization : bool
            whether or not to simulate also polarization maps
        map_dist : pysm.MapDistribution
            Required for partial sky or MPI, see the PySM docs
        verbose : bool
            Control amount of output
        """

        super().__init__(nside=nside, map_dist=map_dist)
        self.maps = {}
        self.maps = self.get_filenames(path)

        self.freqs = np.array(list(self.maps.keys()))
        self.freqs.sort()
        self.input_units = input_units
        self.has_polarization = has_polarization
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

        nu = freqs.to(u.GHz).value
        weights = utils.normalize_weights(freqs, weights)

        if not np.isscalar(nu) and len(nu) == 1:
            nu = nu[0]

        if np.isscalar(nu):

            # special case: we request only 1 frequency and that is among the ones
            # available as input
            check_isclose = np.isclose(self.freqs, nu)
            if np.any(check_isclose):

                freq = self.freqs[check_isclose][0]
                out = self.read_map_by_frequency(freq)
                if self.has_polarization:
                    return out << u.uK_RJ
                else:
                    zeros = np.zeros_like(out)
                    return np.array([out, zeros, zeros]) << u.uK_RJ

            else:  # continue with interpolation as with an array of nus
                nu = np.array([nu])
        else:
            nu = np.asarray(nu)

        assert (
            nu[0] >= self.freqs[0]
        ), "Frequency not supported, requested {} Ghz < lower bound {} GHz".format(
            nu[0], self.freqs[0]
        )
        assert (
            nu[-1] <= self.freqs[-1]
        ), "Frequency not supported, requested {} Ghz > upper bound {} GHz".format(
            nu[-1], self.freqs[-1]
        )

        first_freq_i, last_freq_i = np.searchsorted(self.freqs, [nu[0], nu[-1]])
        first_freq_i -= 1
        last_freq_i += 1

        freq_range = self.freqs[first_freq_i:last_freq_i]

        if self.verbose:
            print("Frequencies considered:", freq_range)

        if self.map_dist is None or self.map_dist.pixel_indices is None:
            npix = hp.nside2npix(self.nside)
        else:
            npix = len(self.map_dist.pixel_indices)

        # allocate a single array for all maps to be used by the interpolator
        # always use size 3 for polarization because PySM always expects IQU maps

        all_maps = np.zeros(
            (len(freq_range), 3 if self.has_polarization else 1, npix), dtype=np.double
        )

        for i, freq in enumerate(freq_range):
            if self.has_polarization:
                all_maps[i] = self.read_map_by_frequency(freq)
                if self.verbose:
                    for i_pol, pol in enumerate("IQU"):
                        print(
                            "Mean emission at {} GHz in {}: {:.4g} uK_RJ".format(
                                freq, pol, all_maps[i][i_pol].mean()
                            )
                        )
            else:
                all_maps[i][0] = self.read_map_by_frequency(freq)

        out = compute_interpolated_emission_numba(nu, weights, freq_range, all_maps)

        # the output of out is always 2D, (IQU, npix)
        return out << u.uK_RJ

    def read_map_by_frequency(self, freq):
        filename = self.maps[freq]
        return self.read_map_file(freq, filename)

    def read_map_file(self, freq, filename):
        if self.verbose:
            print("Reading map {}".format(filename))
        m = self.read_map(
            filename,
            field=(0, 1, 2) if self.has_polarization else 0,
            unit=self.input_units,
        )
        return m.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq*u.GHz)).value


@njit(parallel=False)
def compute_interpolated_emission_numba(freqs, weights, freq_range, all_maps):
    output = np.zeros(all_maps[0].shape, dtype=all_maps.dtype)
    index_range = np.arange(len(freq_range))
    for i in range(len(freqs)):
        interpolation_weight = np.interp(freqs[i], freq_range, index_range)
        int_interpolation_weight = int(interpolation_weight)
        m = (interpolation_weight - int_interpolation_weight) * all_maps[
            int_interpolation_weight
        ]
        m += (int_interpolation_weight + 1 - interpolation_weight) * all_maps[
            int_interpolation_weight + 1
        ]

        trapz_step_inplace(freqs, weights, i, m, output)
    return output
