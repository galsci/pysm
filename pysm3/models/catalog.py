import numpy as np
from numba import njit

# from astropy import constants as const
#
from .. import units as u

# from .. import utils
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
    See the documentation for an example on how to create a catalog
    file with `xarray`.

    Parameters
    ----------
    catalog_filename: str or Path
        Path to the catalog HDF5 file
    """

    def __init__(
        self,
        catalog_filename,
        nside=None,
        target_shape=None,
        target_wcs=None,
        map_dist=None,
    ):
        self.catalog_filename = catalog_filename
        self.nside = nside
        self.shape = target_shape
        self.wcs = target_wcs

        with h5py.File(self.catalog_filename) as f:
            assert f["theta"].attrs["units"].decode("UTF-8") == "rad"
            assert f["phi"].attrs["units"].decode("UTF-8") == "rad"
            assert f["logpolycoefflux"].attrs["units"].decode("UTF-8") == "Jy"
            assert f["logpolycoefpolflux"].attrs["units"].decode("UTF-8") == "Jy"

        assert map_dist is None, "Distributed execution not supported"

    def get_fluxes(self, freqs: u.GHz, coeff="logpolycoefflux", weights=None):
        """Get catalog fluxes integrated over a bandpass"""
        with h5py.File(self.catalog_filename) as f:
            flux = evaluate_model(freqs.to_value(u.GHz), weights, np.array(f[coeff]))
        return flux

    @u.quantity_input
    def get_emission(
        self,
        freqs: u.GHz,
        fwhm: [u.arcmin, None] = None,
        weights=None,
        output_units=u.uK_RJ,
    ):
        raise NotImplemented()
