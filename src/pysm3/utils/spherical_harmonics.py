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
    r"""Apply smoothing and coordinate rotation to an input map

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


def get_differential_fwhm(target_fwhm, pre_applied_beam=None):
    """Return the additional Gaussian FWHM needed to reach a target resolution
    starting from a template that already carries some amount of smoothing.

    Parameters
    ----------
    target_fwhm : astropy.units.Quantity
        Target output FWHM (an angle).
    pre_applied_beam : astropy.units.Quantity, optional
        FWHM of the beam already applied by the input template: either a scalar
        (same presmoothing for all components) or an array of shape ``(3,)``
        (per-component IQU presmoothing). ``None`` or ``0`` means the template
        carries no smoothing, in which case the full ``target_fwhm`` is needed.

    Returns
    -------
    differential_fwhm : astropy.units.Quantity
        Additional FWHM to apply on top of the template, a scalar or an array of
        shape ``(3,)`` matching ``pre_applied_beam``. Returns 0 wherever the
        target is smaller than or equal to the pre-applied beam: a template
        cannot be de-convolved, so no additional smoothing is applied there.

    Notes
    -----
    For Gaussian beams the convolution of two Gaussians is another Gaussian whose
    widths add in quadrature, so the extra smoothing is
    ``sqrt(target_fwhm**2 - pre_applied_beam**2)``. PySM records template
    presmoothing as a Gaussian FWHM (see the ``SMOOTHING_ANGLE`` FITS keyword),
    so this FWHM-based result is exact for those templates.
    """
    target = np.atleast_1d(target_fwhm.to_value(u.radian))
    if pre_applied_beam is None:
        pre = np.zeros_like(target)
    else:
        # u.Quantity(value, rad) also accepts plain numbers (e.g. 0) and
        # strings (e.g. "53 arcmin"), converting them to radians
        pre = np.atleast_1d(u.Quantity(pre_applied_beam, u.radian).to_value(u.radian))
    target, pre = np.broadcast_arrays(target, pre)
    differential = np.sqrt(np.maximum(target**2 - pre**2, 0.0))
    if differential.size == 1:
        differential = differential[0]
    return np.asarray(differential) << u.radian


def get_differential_beam_window(target_fwhm, pre_applied_beam=None, lmax=None):
    """Return the per-component net beam window to apply to reach ``target_fwhm``
    starting from a template that already carries ``pre_applied_beam``.
    This uses the window-division convention, like
    :class:`~pysm3.InterpolatingComponent`: the net window is the ratio of the
    target window to the pre-applied window

    .. math::

        B_\\ell = \\frac{g_\\ell(\\mathrm{target})}{g_\\ell(\\mathrm{pre})}

    For Gaussian beams this is equivalent to smoothing with
    ``sqrt(target**2 - pre**2)`` (see :func:`get_differential_fwhm`), and it is
    computed exactly that way, as the Gaussian beam of the differential FWHM:
    dividing the two windows directly would give ``0/0 = NaN`` where both
    underflow to zero (degree-scale beams at ``lmax`` of a few thousand). The
    window-division convention itself also generalizes to non-Gaussian /
    measured windows, as :class:`~pysm3.InterpolatingComponent` does.
    Wherever the target is
    smaller than or equal to the pre-applied beam the window is set to unity
    (a template cannot be de-convolved, so no change is applied there).

    Note the rows are all built from the spin-0 (``pol=False``) Gaussian
    windows, which is consistent with how the window is applied to each
    component. :class:`~pysm3.InterpolatingComponent` instead divides the
    ``pol=True`` windows, whose E/B rows carry an additional
    ``exp(2*sigma**2)`` spin factor: the two agree exactly for the T row and
    up to a constant ``exp(2*(sigma_target**2 - sigma_pre**2))`` factor for
    E/B, negligible (relative ~1e-5) for the arcminute-to-degree
    differentials this feature targets but reaching ~0.3% for a 5 deg
    differential.

    Parameters
    ----------
    target_fwhm : astropy.units.Quantity
        Target output FWHM (an angle).
    pre_applied_beam : astropy.units.Quantity, optional
        FWHM already applied by the template, scalar or shape ``(3,)``.
        ``None`` or ``0`` means no presmoothing (the target window is returned).
    lmax : int
        Maximum multipole of the window.

    Returns
    -------
    beam_window : np.ndarray
        Array of shape ``(3, lmax+1)`` usable as the ``beam_window`` argument of
        :func:`apply_smoothing_and_coord_transform`.
    """
    if lmax is None:
        msg = "lmax must be provided to build a beam window"
        raise ValueError(msg)
    target = np.atleast_1d(target_fwhm.to_value(u.radian))
    if pre_applied_beam is None or np.all(np.asarray(pre_applied_beam) == 0):
        return np.stack(
            [
                hp.gauss_beam(target[0] if target.size == 1 else target[i], lmax=lmax)
                for i in range(3)
            ]
        )
    # u.Quantity(value, rad) also accepts plain numbers (e.g. 0) and
    # strings (e.g. "53 arcmin"), converting them to radians, like
    # get_differential_fwhm does
    pre = np.atleast_1d(u.Quantity(pre_applied_beam, u.radian).to_value(u.radian))
    net = np.ones((3, lmax + 1))
    for i in range(3):
        t = target[0] if target.size == 1 else target[i]
        b = pre[0] if pre.size == 1 else pre[i]
        if t > b:
            # gauss_beam(t) / gauss_beam(b) == gauss_beam(sqrt(t**2 - b**2))
            # for Gaussian beams; computing the ratio as the beam of the
            # differential FWHM gives exactly 0 (not 0/0 = NaN) where both
            # windows underflow at high lmax
            net[i] = hp.gauss_beam(np.sqrt(t**2 - b**2), lmax=lmax)
    return net


def apply_differential_smoothing(map_t, pre_applied_beam, target_fwhm):
    """Smooth a template map by only the *differential* between the target beam
    and the beam the template already carries.

    This is the reusable building block of the presmoothing feature: it is what
    any template model calls on each of its beam-carrying amplitude maps when a
    ``smoothing_angle`` is requested (see :class:`pysm3.Model`).
    ``pre_applied_beam`` is the presmoothing of this specific map (e.g. one
    element of the model's ``pre_applied_beam``); pass it as ``0`` or ``None``
    for a map with no recorded presmoothing to get a full target smoothing.
    Returns ``map_t`` unchanged when the target does not exceed the
    pre-applied beam (a template cannot be de-convolved).

    Parameters
    ----------
    map_t : astropy.units.Quantity
        The amplitude template map to smooth: 1-D / ``(1, npix)`` with a
        scalar ``pre_applied_beam``, or ``(3, npix)`` with a scalar or
        shape-``(3,)`` ``pre_applied_beam`` (per-component IQU presmoothing).
        A ``(3, npix)`` map is smoothed through the joint TEB transform (the
        IQU convention of
        :func:`apply_smoothing_and_coord_transform`), so that Q / U are
        treated as components of a spin-2 field rather than as independent
        scalar maps.
    pre_applied_beam : astropy.units.Quantity or None
        FWHM already applied to this template, a scalar or, for a
        ``(3, npix)`` map, shape ``(3,)``. ``None``/``0`` means no
        presmoothing.
    target_fwhm : astropy.units.Quantity
        Target output FWHM.

    Returns
    -------
    astropy.units.Quantity
        The map smoothed by the differential (per component when
        ``pre_applied_beam`` has one entry per component).
    """
    differential = get_differential_fwhm(target_fwhm, pre_applied_beam)
    if np.all(differential.value == 0):
        return map_t
    if differential.isscalar:
        return apply_smoothing_and_coord_transform(map_t, fwhm=differential)
    # Per-component presmoothing: apply the net beam window, which is unity
    # for the components whose target does not exceed their presmoothing
    map_shape = np.shape(map_t)
    if len(map_shape) < 2 or map_shape[0] != differential.shape[0]:
        raise ValueError(
            "pre_applied_beam with {} components requires a map with {} "
            "components, got shape {}".format(
                differential.shape[0], differential.shape[0], map_shape
            )
        )
    lmax = int(2.5 * hp.get_nside(map_t))
    beam_window = get_differential_beam_window(target_fwhm, pre_applied_beam, lmax=lmax)
    return apply_smoothing_and_coord_transform(map_t, beam_window=beam_window, lmax=lmax)


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
