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

try:
    import pixell
except ImportError:
    pixell = None

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

    @property
    def includes_smoothing(self):
        return self.pre_applied_beam is not None

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
        output_car_resol=None,
        coord=None,
        lmax: Optional[int] = None,
        return_healpix=True,
        return_car=False,
    ) -> u.Quantity[u.uK_RJ]:
        """Return the emission at a frequency or integrated over a bandpass

        Parameters
        ----------
            freqs : u.Quantity[u.GHz]
                Frequency or frequencies at which to compute the emission
            weights : array-like
                Weights for the bandpass integration
            fwhm : u.Quantity[u.arcmin]
                Beam FWHM
            output_nside : int
                NSIDE of the output map
            coord: tuple of str
                coordinate rotation, it uses the healpy convention, "Q" for Equatorial,
                "G" for Galactic.
            output_car_resol : astropy.Quantity
                CAR output map resolution, generally in arcmin
            lmax : int
                Maximum l of the alm transform
            return_healpix : bool
                If True, return the map in healpix format
            return_car : bool
                If True, return the map in CAR format

        Returns
        -------
            m : u.Quantity[u.uK_RJ]
                Emission at the requested frequency or integrated over the bandpass
        """
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

        pre_beam = (
            None
            if self.pre_applied_beam is None
            else (
                self.pre_applied_beam.get(
                    self.nside, self.pre_applied_beam[str(self.templates_nside)]
                )
                * self.pre_applied_beam_units
            )
        )
        if (
            ((pre_beam is None) or (pre_beam == fwhm))
            and (coord is None)
            and (return_car is False)
        ):
            # no need to go to alm if no beam and no rotation
            output_maps = [out << u.uK_RJ]
        else:

            assert lmax is not None, "lmax must be provided when applying a beam"
            alm = map2alm(out, self.nside, lmax)
            if pre_beam != fwhm:
                log.info(
                    "Applying the differential beam between: %s %s",
                    str(pre_beam),
                    str(fwhm),
                )

                beam = hp.gauss_beam(
                    fwhm.to_value(u.radian), lmax=lmax, pol=True
                ) / hp.gauss_beam(
                    pre_beam.to_value(u.radian),
                    lmax=lmax,
                    pol=True,
                )
                for each_alm, each_beam in zip(alm, beam.T):
                    hp.almxfl(each_alm, each_beam, mmax=lmax, inplace=True)
            if coord is not None:
                rot = hp.Rotator(coord=coord)
                alm = rot.rotate_alm(alm)
            output_maps = []
            if return_healpix:
                log.info("Alm to map HEALPix")
                output_maps.append(
                    hp.alm2map(
                        alm,
                        nside=output_nside if output_nside is not None else self.nside,
                        pixwin=False,
                    )
                    << u.uK_RJ
                )
            if return_car:
                log.info("Alm to map CAR")
                shape, wcs = pixell.enmap.fullsky_geometry(
                    output_car_resol.to_value(u.radian),
                    dims=(3,),
                    variant="fejer1",
                )
                ainfo = pixell.curvedsky.alm_info(lmax=lmax)
                output_maps.append(
                    pixell.curvedsky.alm2map(
                        alm, pixell.enmap.empty(shape, wcs), ainfo=ainfo
                    )
                )

        return output_maps[0] if len(output_maps) == 1 else tuple(output_maps)

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
    output = np.zeros(all_maps[freq_range[0]].shape, dtype=np.float64)
    temp = np.zeros_like(output) if len(freqs) > 1 else output
    index_range = np.arange(len(freq_range))
    for i in range(len(freqs)):
        interpolation_weight = np.interp(freqs[i], freq_range, index_range)
        int_interpolation_weight = int(interpolation_weight)
        relative_weight = np.float64(interpolation_weight - int_interpolation_weight)
        temp[:] = (1 - relative_weight) * all_maps[freq_range[int_interpolation_weight]]
        if relative_weight > 0:
            temp += relative_weight * all_maps[freq_range[int_interpolation_weight + 1]]

        if len(freqs) > 1:
            trapz_step_inplace(freqs, weights, i, temp, output)
    return output
