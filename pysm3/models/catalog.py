import numpy as np
import healpy as hp
from numba import njit
from .. import utils


# from astropy import constants as const
#
from .. import units as u

from .template import Model

import h5py


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
    return flux / (2 * np.pi * sigma**2)


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
            out[i_source] = np.trapz(flux * weights, x=freqs)
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

    def __init__(
        self,
        catalog_filename,
        nside=None,
        target_wcs=None,
        map_dist=None,
    ):
        self.catalog_filename = catalog_filename
        self.nside = nside
        self.shape = (3, hp.nside2npix(nside))
        self.wcs = target_wcs

        with h5py.File(self.catalog_filename) as f:
            assert f["theta"].attrs["units"].decode("UTF-8") == "rad"
            assert f["phi"].attrs["units"].decode("UTF-8") == "rad"
            assert f["logpolycoefflux"].attrs["units"].decode("UTF-8") == "Jy"
            assert f["logpolycoefpolflux"].attrs["units"].decode("UTF-8") == "Jy"

        assert map_dist is None, "Distributed execution not supported"

    def get_fluxes(self, freqs: u.GHz, coeff="logpolycoefflux", weights=None):
        """Get catalog fluxes in Jy integrated over a bandpass"""
        weights /= np.trapz(weights, x=freqs.to_value(u.GHz))
        with h5py.File(self.catalog_filename) as f:
            flux = evaluate_model(freqs.to_value(u.GHz), weights, np.array(f[coeff]))
        return flux * u.Jy

    @u.quantity_input
    def get_emission(
        self,
        freqs: u.GHz,
        fwhm: [u.arcmin, None] = None,
        weights=None,
        output_units=u.uK_RJ,
        car_map_resolution: [u.arcmin, None] = None,
        return_car=False,
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
            it is set to half of the resolution of the HEALPix map given by `self.nside`
        return_car: bool
            If True return a CAR map, if False return a HEALPix map

        Returns
        -------
        output_map: np.array
            Output HEALPix or CAR map"""
        with h5py.File(self.catalog_filename) as f:
            pix = hp.ang2pix(self.nside, f["theta"], f["phi"])
        scaling_factor = utils.bandpass_unit_conversion(
            freqs, weights, output_unit=output_units, input_unit=u.Jy / u.sr
        )
        pix_size = hp.nside2pixarea(self.nside) * u.sr
        if car_map_resolution is None:
            car_map_resolution = (hp.nside2resol(self.nside) * u.rad) / 2

        # Make sure the resolution evenly divides the map vertically
        if (car_map_resolution.to_value(u.rad) % np.pi) > 1e-8:
            car_map_resolution = (
                np.pi / np.round(np.pi / car_map_resolution.to_value(u.rad))
            ) * u.rad
        fluxes_I = self.get_fluxes(freqs, weights=weights, coeff="logpolycoefflux")

        if fwhm is None:
            output_map = np.zeros(self.shape, dtype=np.float32) * output_units
            # sum, what if we have 2 sources on the same pixel?
            output_map[0, pix] += fluxes_I / pix_size * scaling_factor
        else:

            from pixell import (
                enmap,
                pointsrcs,
            )

            shape, wcs = enmap.fullsky_geometry(
                car_map_resolution.to_value(u.radian),
                dims=(3,),
                variant="fejer1",
            )
            output_map = enmap.enmap(np.zeros(shape, dtype=np.float32), wcs)
            r, p = pointsrcs.expand_beam(fwhm2sigma(fwhm.to_value(u.rad)))
            with h5py.File(self.catalog_filename) as f:
                pointing = np.column_stack(
                    (np.pi / 2 - np.array(f["theta"]), np.array(f["phi"]))
                )
            output_map[0] = pointsrcs.sim_objects(
                shape,
                wcs,
                pointing,
                flux2amp(
                    fluxes_I.to_value(u.Jy) * scaling_factor.value,
                    fwhm.to_value(u.rad),
                ),  # to peak amplitude and to output units
                ((r, p)),
            )

        del fluxes_I
        fluxes_P = self.get_fluxes(freqs, weights=weights, coeff="logpolycoefpolflux")
        # set seed so that the polarization angle is always the same for each run
        # could expose to the interface if useful
        np.random.seed(56567)
        psirand = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=len(fluxes_P)
        )
        if fwhm is None:
            output_map[1, pix] += (
                fluxes_P / pix_size * scaling_factor * np.cos(2 * psirand)
            )
            output_map[2, pix] += (
                fluxes_P / pix_size * scaling_factor * np.sin(2 * psirand)
            )
        else:
            pols = [(1, np.cos)]
            pols.append((2, np.sin))
            for i_pol, sincos in pols:
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
        return output_map
