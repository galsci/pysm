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
    ):
        with h5py.File(self.catalog_filename) as f:
            pix = hp.ang2pix(self.nside, f["theta"], f["phi"])
        scaling_factor = utils.bandpass_unit_conversion(
            freqs, weights, output_unit=output_units, input_unit=u.Jy / u.sr
        )
        pix_size = hp.nside2resol(self.nside) * u.sr
        surface_brightness_I = (
            self.get_fluxes(freqs, weights=weights, coeff="logpolycoefflux")
            / pix_size
            * scaling_factor
        )
        output_map = np.zeros(self.shape, dtype=np.float32) * output_units
        if fwhm is None:
            # sum, what if we have 2 sources on the same pixel?
            output_map[0, pix] += surface_brightness_I
        del surface_brightness_I
        surface_brightness_P = (
            self.get_fluxes(freqs, weights=weights, coeff="logpolycoefpolflux")
            / pix_size
            * scaling_factor
        )
        # set seed so that the polarization angle is always the same for each run
        # could expose to the interface if useful
        np.random.seed(56567)
        psirand = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=len(surface_brightness_P)
        )
        if fwhm is None:
            output_map[1, pix] += surface_brightness_P * np.cos(2 * psirand)
            output_map[2, pix] += surface_brightness_P * np.sin(2 * psirand)
        return output_map
