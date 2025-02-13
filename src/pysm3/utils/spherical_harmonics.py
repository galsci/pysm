import healpy as hp
import numpy as np
import logging

from astropy import units as u

try:
    import pixell.enmap, pixell.curvedsky
except ImportError:
    pixell = None

from .. import mpi, utils

log = logging.getLogger("pysm3")


def apply_smoothing_and_coord_transform(
    input_map,
    fwhm=None,
    beam_window=None,
    rot=None,
    lmax=None,
    output_nside=None,
    output_car_resol=None,
    return_healpix=True,
    return_car=False,
    input_alm=False,
    map2alm_lsq_maxiter=None,
    map_dist=None,
):
    """Apply smoothing and coordinate rotation to an input map

    it applies the `healpy.smoothing` Gaussian smoothing kernel if `map_dist`
    is None, otherwise applies distributed smoothing with `libsharp`.
    In the distributed case, no rotation is supported.

    Parameters
    ----------
    input_map : ndarray
        Input map, of shape `(3, npix)`
        This is assumed to have no beam at this point, as the
        simulated small scale template on which the simulations are based
        have no beam.
    fwhm : astropy.units.Quantity
        Full width at half-maximum, defining the
        Gaussian kernels to be applied.
    beam_window: array, optional
        Custom beam window function (:math:`B_\ell`)
    rot: hp.Rotator
        Apply a coordinate rotation give a healpy `Rotator`, e.g. if the
        inputs are in Galactic, `hp.Rotator(coord=("G", "C"))` rotates
        to Equatorial
    output_nside : int
        HEALPix output map Nside, if None, use the same as the input
    lmax : int
        lmax for the map2alm step, if None, it is set to 2.5 * nside
        if output_nside is equal or higher than nside.
        It is set to 1.5 * nside if output_nside is lower than nside
    output_car_resol : astropy.Quantity
        CAR output map resolution, generally in arcmin
    return_healpix : bool
        Whether to return the HEALPix map
    return_car : bool
        Whether to return the CAR map
    input_alm : bool
        Instead of starting from a map, `input_map` is a set of Alm
    map2alm_lsq_maxiter : int
        Number of iteration for the least squares map to Alm transform,
        setting it to 0 uses the standard map2alm, the default of 10
        makes the transform slow if the input map is not band limited,
        for example if has point sources or sharp features.
        If ell_max is <= 1.5 nside, this setting is ignored
        and `map2alm` with pixel weights is used.

    Returns
    -------
    smoothed_map : np.ndarray or tuple of np.ndarray
        Array containing the smoothed sky or tuple of HEALPix and CAR maps
    """

    if not input_alm:
        nside = hp.get_nside(input_map)
        if output_nside is None:
            output_nside = nside

    unit = input_map.unit if hasattr(input_map, "unit") else 1

    if lmax is None:
        if nside == output_nside:
            lmax = int(2.5 * output_nside)
        elif output_nside > nside:
            lmax = int(2.5 * nside)
        elif output_nside < nside:
            lmax = int(1.5 * nside)
        log.info("Setting lmax to %d", lmax)

    output_maps = []

    if map_dist is None:
        if input_alm:
            alm = input_map.copy()
        else:
            alm = map2alm(input_map, nside, lmax, map2alm_lsq_maxiter)
        if fwhm is not None:
            assert beam_window is None, "Either FWHM or beam_window"
            log.info("Smoothing with fwhm of %s", str(fwhm))
            hp.smoothalm(alm, fwhm=fwhm.to_value(u.rad), inplace=True, pol=True)
        if beam_window is not None:
            assert fwhm is None, "Either FWHM or beam_window"
            log.info("Smoothing with a custom isotropic beam")
            # smoothalm does not support polarized beam
            for i in range(3):
                try:
                    beam_window_i = beam_window[i, :]
                    log.info("Using polarized beam")
                except IndexError:
                    beam_window_i = beam_window
                    log.info("Using the same beam for all components")
                hp.smoothalm(alm[i], beam_window=beam_window_i, inplace=True)
        if rot is not None:
            log.info("Rotate Alm")
            rot.rotate_alm(alm, inplace=True)
        if return_healpix:
            log.info("Alm to map HEALPix")
            if input_alm:
                assert (
                    output_nside is not None
                ), "If inputting Alms, specify output_nside"
            output_maps.append(
                u.Quantity(
                    hp.alm2map(alm, nside=output_nside, pixwin=False), unit, copy=False
                )
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
    else:
        assert (rot is None) or (
            rot.coordin == rot.coordout
        ), "No rotation supported in distributed smoothing"
        output_maps.append(mpi.mpi_smoothing(input_map, fwhm, map_dist))
        assert not return_car, "No CAR output supported in Libsharp smoothing"

    return output_maps[0] if len(output_maps) == 1 else tuple(output_maps)


def map2alm(input_map, nside, lmax, map2alm_lsq_maxiter=None):
    """Compute alm from a map using healpy.

    Automatically selects the most appropriate method based on
    the target lmax

    Parameters
    ----------
    input_map : np.ndarray
        Input HEALPix map
    nside : int
        Resolution parameter of the input map
    lmax : int
        Maximum multipole of the alm
    map2alm_lsq_maxiter : int, optional
        Maximum number of iterations for map2alm_lsq, by default 10

    Returns
    -------
    alm: np.ndarray
        alm array"""
    if map2alm_lsq_maxiter is None:
        map2alm_lsq_maxiter = 10
    nside = hp.get_nside(input_map)
    if lmax <= 1.5 * nside:
        log.info("Using map2alm with pixel weights")
        alm = hp.map2alm(
            input_map,
            lmax=lmax,
            use_pixel_weights=True if nside > 16 else False,
        )
    elif map2alm_lsq_maxiter == 0:
        alm = hp.map2alm(input_map, lmax=lmax, iter=0)
        log.info("Using map2alm with no weights and no iterations")
    else:
        alm, error, n_iter = hp.map2alm_lsq(
            input_map,
            lmax=lmax,
            mmax=lmax,
            tol=1e-7,
            maxiter=map2alm_lsq_maxiter,
        )
        if n_iter == map2alm_lsq_maxiter:
            log.warning(
                "hp.map2alm_lsq did not converge in %d iterations,"
                + " residual relative error is %.2g",
                n_iter,
                error,
            )
        else:
            log.info(
                "Used map2alm_lsq, converged in %d iterations,"
                + "residual relative error %.2g",
                n_iter,
                error,
            )
    return alm
