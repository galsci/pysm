import logging
import os
from typing import Optional
import warnings

import healpy as hp
import numpy as np
from numba import njit, types
from numba.typed import Dict

from .. import units as u
from .. import utils
from ..utils import trapz_step_inplace, map2alm
from .template import Model

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
        freqs=None,
        available_nside=None,
        pre_applied_beam=None,
        pre_applied_beam_units=None,
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

        super().__init__(
            nside=nside,
            available_nside=available_nside,
            max_nside=max_nside,
            map_dist=map_dist,
        )
        if freqs is None:
            self.maps = {}
            self.maps = self.get_filenames(path)
            self.freqs = np.array(list(self.maps.keys()))
            self.freqs.sort()
        else:
            self.freqs = np.array(freqs)
            self.maps = {freq: path + f"/{freq:05.1f}.fits" for freq in freqs}

        self.pre_applied_beam = pre_applied_beam
        if pre_applied_beam_units is not None:
            self.pre_applied_beam_units = u.Unit(pre_applied_beam_units)
        # use a numba typed Dict so we can used in JIT compiled code
        self.cached_maps = Dict.empty(
            key_type=types.float64, value_type=types.float32[:, :]
        )

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
    def get_emission(
        self,
        freqs: u.Quantity[u.GHz],
        weights=None,
        fwhm: Optional[u.Quantity[u.arcmin]] = None,
        output_nside: Optional[int] = None,
        lmax: Optional[int] = None,
    ) -> u.Quantity[u.uK_RJ]:
        nu = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(nu, weights)

        # special case: we request only 1 frequency and that is among the ones
        # available as input
        check_isclose = np.isclose(self.freqs, nu[0])
        if len(nu) == 1 and np.any(check_isclose):
            freq = self.freqs[check_isclose][0]
            log.info("Selecting single frequency map: %s", str(freq))
            out = self.read_map_by_frequency(freq)
            if out.ndim == 1 or out.shape[0] == 1:
                zeros = np.zeros_like(out)
                out = np.array([out, zeros, zeros])
        else:

            npix = hp.nside2npix(self.nside)
            if nu[0] < self.freqs[0]:
                warnings.warn(
                    f"Frequency not supported, requested {nu[0]} Ghz < lower bound {self.freqs[0]} GHz"
                )
                return np.zeros((3, npix)) << u.uK_RJ
            if nu[-1] > self.freqs[-1]:
                warnings.warn(
                    f"Frequency not supported, requested {nu[-1]} Ghz > upper bound {self.freqs[-1]} GHz"
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
                            f"Mean emission at {freq} GHz in {pol}: {self.cached_maps[freq][i_pol].mean():.4g} uK_RJ"
                        )

            out = compute_interpolated_emission_numba(
                nu, weights, freq_range, self.cached_maps
            )

            if out.ndim == 1 or out.shape[0] == 1:
                if out.ndim == 2:
                    out = out[0]
                zeros = np.zeros_like(out)
                out = np.array([out, zeros, zeros])

        if (self.pre_applied_beam is not None) and (fwhm is not None):
            assert lmax is not None, "lmax must be provided when applying a beam"
            if output_nside is None:
                output_nside = self.nside
            pre_beam = (
                self.pre_applied_beam.get(
                    self.nside, self.pre_applied_beam[self.available_nside[0]]
                )
                * self.pre_applied_beam_units
            )
            if pre_beam != fwhm:
                log.info(
                    "Applying the differential beam between: %s %s",
                    str(pre_beam),
                    str(fwhm),
                )
                alm = map2alm(out, self.nside, lmax)

                beam = hp.gauss_beam(
                    fwhm.to_value(u.radian), lmax=lmax, pol=True
                ) / hp.gauss_beam(
                    pre_beam.to_value(u.radian),
                    lmax=lmax,
                    pol=True,
                )
                for each_alm, each_beam in zip(alm, beam.T):
                    hp.almxfl(each_alm, each_beam, mmax=lmax, inplace=True)
                out = hp.alm2map(alm, nside=output_nside, pixwin=False)

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
    temp = np.zeros_like(output) if len(freqs) > 1 else output
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
