from typing import Optional
import h5py
import healpy as hp
import numpy as np

import logging

log = logging.getLogger("pysm3")

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
from numba import njit

# from astropy import constants as const
#
from .. import units as u
from .. import utils
from .template import Model


@njit
def aggregate(index, array, values):
    """Sums values by index

    Example:
    m = np.zeros(3)
    m[2, 2] += np.ones(2)
    gives:
    m
    [0, 0, 1]
    instead
    aggregate([2,2], m, np.ones(2))
    gives
    m
    [0, 0, 2]
    """
    for i, v in zip(index, values):
        array[i] += v


@njit
def fwhm2sigma(fwhm):
    """Converts the Full Width Half Maximum of a Gaussian beam to its standard deviation"""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@njit
def flux2amp(flux, fwhm):
    """Converts the total flux of a radio source to the peak amplitude of its Gaussian
    beam representation, taking into account the width of the beam as specified
    by its FWHM

    Parameters
    ----------
    flux: float
        Total flux of the radio source
    fwhm: float
        Full Width Half Maximum of the beam in radians

    Returns
    -------
    amp: float
        Peak amplitude of the Gaussian beam representation of the radio source"""
    sigma = fwhm2sigma(fwhm)
    amp = flux / (2 * np.pi * sigma**2)
    # sim_objects fails if amp is zero
    c = 1e-8  # clip
    amp[np.logical_and(amp < c, amp >= 0)] = c
    amp[np.logical_and(amp > -c, amp < 0)] = -c
    return amp


@njit
def evaluate_poly(p, x):
    """Low level polynomial evaluation, both input are 1D
    same interface of np.polyval.
    Having this implemented in numba should allow numba
    to provide better optimization. If not, just use
    np.polyval directly."""

    out = 0
    N = len(p)
    for i in range(N):
        out += p[i] * x ** (N - 1 - i)
    out = max(0, out)
    return out


@njit
def evaluate_model(freqs, weights, coeff):
    """Integrate log polynomial model across the bandpass for
    each source in the catalog

    Parameters
    ----------
    freqs: np.array
        Array of frequencies in GHz
    weights: np.array
        Array of relative bandpass weights already normalized
        Same length of freqs
    coeff: 2D np.array (n_sources, n_coeff)
        Array of log polynomial coefficients for each source

    Returns
    -------
    flux: np.array
        Array of the flux of each source integrated over the band
    """
    n_sources = coeff.shape[0]
    logfreqs = np.log(freqs)
    out = np.zeros(n_sources, dtype=np.float64)
    assert len(freqs) == len(weights)
    if len(freqs) == 1:
        for i_source in range(n_sources):
            out[i_source] = evaluate_poly(coeff[i_source, :], logfreqs[0])
    else:
        flux = np.zeros(len(freqs), dtype=np.float64)
        for i_source in range(n_sources):
            for i_freq in range(len(freqs)):
                flux[i_freq] = evaluate_poly(coeff[i_source, :], logfreqs[i_freq])
            out[i_source] = trapezoid(flux * weights, x=freqs)
    return out


class PointSourceCatalog(Model):
    """Model for a Catalog of point sources defined with their coordinates and
    a model of their emission based on a logpolynomial of frequency.
    The beam convolution is performed in map domain with `pixell`.

    The catalog should be in HDF5 format, with the fields:
    theta: colatitude in radians
    phi: longitude in radians
    logpolycoefflux and logpolycoefpolflux: polynomial coefficients in natural
    logaritm (`np.log`) of the frequency, typically 4th order, but accepts
    any order. (source_index, pol_order). Unit needs to be Jy
    each field should have an attribute units which is checked when loading
    a model. No conversion is performed.
    See the documentation and the unit tests for examples on how to create a
    catalog file with `xarray`.

    Parameters
    ----------
    catalog_filename: str or Path
        Path to the catalog HDF5 file
    """

    includes_smoothing = True

    def __init__(
        self,
        catalog_filename,
        nside=None,
        target_wcs=None,
        catalog_slice=None,
        max_nside=None,
        map_dist=None,
    ):
        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        self.catalog_filename = utils.RemoteData().get(catalog_filename)
        self.wcs = target_wcs
        if catalog_slice is None:
            catalog_slice = np.index_exp[:]
        self.catalog_slice = catalog_slice

        with h5py.File(self.catalog_filename) as f:
            assert f["theta"].attrs["units"].decode("UTF-8") == "rad"
            assert f["phi"].attrs["units"].decode("UTF-8") == "rad"
            assert f["logpolycoefflux"].attrs["units"].decode("UTF-8") == "Jy"
            assert f["logpolycoefpolflux"].attrs["units"].decode("UTF-8") == "Jy"

        assert map_dist is None, "Distributed execution not supported"

    def get_fluxes(self, freqs: u.GHz, coeff="logpolycoefflux", weights=None):
        """Get catalog fluxes in Jy integrated over a bandpass"""
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        with h5py.File(self.catalog_filename) as f:
            flux = evaluate_model(
                freqs, weights, np.array(f[coeff][self.catalog_slice])
            )
        return flux * u.Jy

    @u.quantity_input
    def get_emission(
        self,
        freqs: u.Quantity[u.GHz],
        fwhm: Optional[u.Quantity[u.arcmin]] = None,
        weights=None,
        output_units=u.uK_RJ,
        car_map_resolution: Optional[u.Quantity[u.arcmin]] = None,
        lmax=None,
        output_nside=None,
        coord=None,
        return_car=False,
        output_car_resol=None,
        return_healpix=True,
    ):
        """Generate a HEALPix or CAR map of the catalog emission integrated on the bandpass
        and convolved with the beam

        Parameters
        ----------
        freqs: np.array
            Array of frequencies in GHz
        fwhm: float or None
            Full Width Half Maximum of the beam in arcminutes, if None, each source is assigned
            to a single pixel
        weights: np.array
            Array of relative bandpass weights already normalized
            Same length of freqs, if None, uniform weights are assumed
        output_units: astropy.units
            Output units of the map
        car_map_resolution: float
            Resolution of the CAR map used by pixell to generate the map, if None,
            it is set to half of the resolution of the HEALPix map given by `self.nside`, unless
            Nside is above >= 8192, then it is the same resolution of the HEALPix map
        output_nside : int
            Output HEALPix nside, if None, the nside used when creating the component (self.nside)
        lmax: int
            Not used, all operations are performed in pixel domain
        coord: str
            Coordinate rotation to apply, for example ("G", "C") to rotate from Galactic to
            Equatorial coordinates. If None, no rotation is applied
        return_car: bool
            If True return a CAR map
        output_car_resol: arcmin
            Not implemented yet
        return_healpix: bool
            If True return a HEALPix map

        Returns
        -------
        output_map: np.array
            Output HEALPix or CAR map or tuple with HEALPix and CAR maps"""

        convolve_beam = fwhm is not None
        scaling_factor = utils.bandpass_unit_conversion(
            freqs, weights, output_unit=output_units, input_unit=u.Jy / u.sr
        )
        log.info(
            "HEALPix map resolution: %.2e arcmin, nside %d",
            hp.nside2resol(self.nside, arcmin=True),
            self.nside,
        )
        pix_size = hp.nside2pixarea(self.nside) * u.sr

        if return_car:
            assert output_car_resol is None, "CAR resolution not implemented yet"

        if convolve_beam:
            if car_map_resolution is None:
                car_map_resolution = hp.nside2resol(self.nside) * u.rad
                if self.nside < 8192:
                    car_map_resolution /= 2
                log.info(
                    "Rounded CAR map resolution: %s",
                    car_map_resolution.to(u.arcmin).to_string(precision=2),
                )

            # Make sure the resolution evenly divides the map vertically
            if (car_map_resolution.to_value(u.rad) % np.pi) > 1e-8:
                car_map_resolution = (
                    np.pi / np.round(np.pi / car_map_resolution.to_value(u.rad))
                ) * u.rad
                log.info(
                    "Rounded CAR map resolution: %s",
                    car_map_resolution.to(u.arcmin).to_string(precision=2),
                )

        log.info("Computing fluxes for I")
        fluxes_I = self.get_fluxes(freqs, weights=weights, coeff="logpolycoefflux")
        log.info("Fluxes for I computed for %.2f million sources", len(fluxes_I) / 1e6)

        if convolve_beam:
            from pixell import (
                enmap,
                pointsrcs,
            )

            shape, wcs = enmap.fullsky_geometry(
                car_map_resolution.to_value(u.radian),
                dims=(3,),
                variant="fejer1",
            )
            log.info("CAR map shape %s", shape)
            output_map = enmap.enmap(np.zeros(shape, dtype=np.float32), wcs)
            r, p = pointsrcs.expand_beam(fwhm2sigma(fwhm.to_value(u.rad)))

            log.info("Reading pointing")
            with h5py.File(self.catalog_filename) as f:
                pointing = np.vstack(
                    (
                        np.pi / 2 - np.array(f["theta"][self.catalog_slice]),
                        np.array(f["phi"][self.catalog_slice]),
                    )
                )
            log.info("Reading pointing completed")

            amps = flux2amp(
                fluxes_I.to_value(u.Jy) * scaling_factor.value,
                fwhm.to_value(u.rad),
            )  # to peak amplitude and to output units
            log.info("Executing sim_objects for I")
            # import pickle

            # with open(
            #     "sim_objects_inputs.pkl",
            #     "wb",
            # ) as f:
            #     pickle.dump(
            #         {
            #             "shape": shape,
            #             "wcs": wcs,
            #             "poss": pointing,
            #             "amps": amps,
            #             "profile": (r, p),
            #         },
            #         f,
            #     )
            # import sys

            # sys.exit(0)
            output_map[0] = pointsrcs.sim_objects(
                shape=shape,
                wcs=wcs,
                poss=pointing,
                amps=amps,
                profile=((r, p)),
            )
            log.info("Execution of sim_objects for I completed")
        else:
            with h5py.File(self.catalog_filename) as f:
                pix = hp.ang2pix(
                    self.nside,
                    f["theta"][self.catalog_slice],
                    f["phi"][self.catalog_slice],
                )
            output_map = (
                np.zeros((3, hp.nside2npix(self.nside)), dtype=np.float32)
                * output_units
            )
            aggregate(pix, output_map[0], fluxes_I / pix_size * scaling_factor)

        del fluxes_I
        log.info("Computing fluxes for Q/U")
        fluxes_P = self.get_fluxes(freqs, weights=weights, coeff="logpolycoefpolflux")
        log.info(
            "Fluxes for Q/U computed for %.2f million sources", len(fluxes_P) / 1e6
        )
        # set seed so that the polarization angle is always the same for each run
        # could expose to the interface if useful
        np.random.seed(56567)
        psirand = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=len(fluxes_P)
        )
        if convolve_beam:
            pols = [(1, np.cos)]
            pols.append((2, np.sin))
            for i_pol, sincos in pols:
                log.info("Executing sim_objects for Q/U")
                output_map[i_pol] = pointsrcs.sim_objects(
                    shape,
                    wcs,
                    pointing,
                    flux2amp(
                        fluxes_P.to_value(u.Jy)
                        * scaling_factor.value
                        * sincos(2 * psirand),
                        fwhm.to_value(u.rad),
                    ),
                    ((r, p)),
                )
                log.info("Execution of sim_objects for Q/U completed")
            if return_car:
                assert (
                    coord is None
                ), "Coord rotation for CAR not implemented yet, open issue if you need it"
            if return_healpix:
                from pixell import reproject

                frames_dict = {"Q": "equ", "C": "equ", "G": "gal"}
                if coord is not None:
                    coord = ",".join([frames_dict[frame] for frame in coord])

                log.info("Reprojecting to HEALPix")
                output_nside = output_nside or self.nside
                output_map_healpix = (
                    reproject.map2healpix(
                        output_map, output_nside, rot=coord, method="spline"
                    )
                    * output_units
                )
                log.info("Reprojecting to HEALPix completed")
                if return_car:
                    output_map = (output_map_healpix, output_map)
                else:
                    output_map = output_map_healpix

        else:
            assert not return_car and return_healpix, "Only HEALPix maps supported"
            aggregate(
                pix,
                output_map[1],
                fluxes_P / pix_size * scaling_factor * np.cos(2 * psirand),
            )
            aggregate(
                pix,
                output_map[2],
                fluxes_P / pix_size * scaling_factor * np.sin(2 * psirand),
            )
        log.info("Catalog emission computed")
        return output_map
