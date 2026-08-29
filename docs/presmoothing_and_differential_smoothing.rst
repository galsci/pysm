.. _presmoothing-and-differential-smoothing:

Presmoothing and Differential Smoothing
***************************************

Many PySM input templates are not point-like: they are delivered to us already
convolved with a beam of a known size, because the data they are built from
(e.g. Haslam 408 MHz, WMAP 9-year, or Planck) have a native resolution of tens
of arcminutes to a degree.

If you request an output map at some target resolution :math:`R` you do *not*
generally want to convolve an already-smoothed template by a fresh full
:math:`R` beam -- that would *double apply* part of the beam and over-smooth the
result. Instead, PySM can now treat the template's built-in beam as a
first-class property, read it from the template's FITS header, and apply only
the **differential** smoothing needed to go from the resolution the template
already has up to the requested :math:`R`.

This page documents the feature end to end: how a template records its
presmoothing, how models expose it, and how to ask PySM for an output at a given
resolution.

The motivating example
======================

A synchrotron template built from the WMAP-K band already carries a beam of
about ``53 arcmin``. If a user requests an output map at ``1 deg``, the correct
result is obtained by smoothing the template by only

.. math::

    \sqrt{(1\deg)^2 - (53\arcmin)^2} \approx 0.47\deg

on top -- not by a fresh full ``1 deg`` convolution. The same logic applies to
each component independently, so intensity (often from Haslam, ``56 arcmin``)
and polarization (often from WMAP, ``53 arcmin``) can carry *different*
presmoothing and still both land exactly at the requested resolution.

How a template records its presmoothing
=======================================

Presmoothing is a property of the *template file*, not of the model. It is
stored as a Gaussian FWHM in the template's FITS header under the
``SMOOTHING_ANGLE`` keyword, written with :func:`pysm3.utils.add_metadata`::

    from pysm3.utils import add_metadata
    add_metadata([file_name], coord="G", unit="uK_RJ", smoothing_angle="53 arcmin")

The value is any string parseable by :mod:`astropy.units` (e.g. ``"1 deg"``,
``"53 arcmin"``). A template without the keyword (or with a value that cannot be
parsed) is treated as *unsmoothed* (presmoothing ``0``), preserving the
existing behaviour of every current PySM preset.

When ``read_map`` loads such a template it exposes the recorded angle as the
``smoothing_angle`` attribute of the returned map, and
:func:`pysm3.extract_smoothing_angle` reads it directly from a file::

    import pysm3
    m = pysm3.read_map("template.fits", nside=512, field=[0, 1, 2])
    print(m.smoothing_angle)          # 53.0 arcmin

Exposing presmoothing on a model
================================

Models that load amplitude templates aggregate the per-component presmoothing
into a single ``pre_applied_beam`` attribute -- a scalar or shape-``(3,)``
:class:`astropy.units.Quantity` holding the FWHM already applied to
I / Q / U respectively. The :class:`pysm3.Model` base class initializes it to
``None`` and provides a convenience property::

    model.pre_applied_beam     # Quantity: beam already applied to I, Q, U (e.g. [56., 53., 53.] arcmin)
    model.includes_smoothing   # True if any component carries presmoothing

``includes_smoothing`` is the "consumer" of this feature: any code can ask "does
this component/sky already include its beam, so I should not re-smooth it?".

Asking PySM for an output at a target resolution
================================================

The user-facing entry point is a new ``smoothing_angle`` argument accepted by
:class:`pysm3.Sky` and forwarded to the template components that support it
(e.g. :class:`pysm3.PowerLaw` and :class:`pysm3.CurvedPowerLaw`)::

    import astropy.units as u
    import pysm3

    sky = pysm3.Sky(
        nside=2048,
        component_config={
            "my_synch": {
                "class": "PowerLaw",
                "map_I": "my_presmoothed_I.fits",
                "map_Q": "my_presmoothed_Q.fits",
                "map_U": "my_presmoothed_U.fits",
                "freq_ref_I": "23 GHz",
                "map_pl_index": -3.0,
            }
        },
        smoothing_angle=1 * u.deg,
    )
    m = sky.get_emission(23 * u.GHz)

The amplitude templates and their spectral parameters are passed exactly as
usual; the only extra step is that each amplitude template must carry a
``SMOOTHING_ANGLE`` header (see above) recording the beam it already has. At
model construction time each template is smoothed by only the *differential*
between ``smoothing_angle`` and its presmoothing, and the emission is then
computed through the normal spectral law.

The first model planned to make use of this is the IP2026 low-frequency
synchrotron model (``s8``); once its templates are shipped with the
``SMOOTHING_ANGLE`` header, ``Sky(preset_strings=["s8"], smoothing_angle=1*u.deg)``
will work out of the box.

Two equivalent conventions
==========================

For a Gaussian presmoothing of FWHM :math:`b` and a target :math:`R`, the extra
smoothing can be computed two equivalent ways (they coincide exactly for
Gaussian beams):

* **FWHM quadrature** -- smooth by the differential FWHM

  .. math:: \Delta = \sqrt{\max(R^2 - b^2,\; 0)},

  implemented by :func:`pysm3.get_differential_fwhm`. This is what the
  build-time model smoothing uses, and it is exact for the Gaussian
  presmoothing recorded by PySM templates.

* **Window division** -- divide the target window by the pre-applied window,

  .. math:: B_\ell = \frac{g_\ell(R)}{g_\ell(b)},

  implemented by :func:`pysm3.get_differential_beam_window`. This stays exact
  even for non-Gaussian / measured windows and is the same convention already
  used by :class:`pysm3.InterpolatingComponent`.

Both helpers return ``0`` / identity when :math:`R \le b` -- a template cannot be
*de*convolved, so no additional smoothing (and no "de-smoothing") is applied in
that case. The returned per-component window has shape ``(3, lmax+1)`` and can be
passed directly as the ``beam_window`` argument of
:func:`pysm3.apply_smoothing_and_coord_transform`.

Templates with no recorded presmoothing
=======================================

For every existing preset template (``d1``, ``s1``, ``s7``, ...) the
``SMOOTHING_ANGLE`` keyword is absent, so ``pre_applied_beam == 0`` and the full
target :math:`R` is applied. Setting ``smoothing_angle`` on such a sky is
therefore exactly equivalent to the current unsmoothed behaviour plus one
smoothing by :math:`R`. **Backwards compatibility is preserved.**

Resolution floor
================

If the requested ``smoothing_angle`` is *smaller* than a template's native
presmoothing, the model cannot deliver finer than that template; the behavior is
to leave the template unchanged (no de-convolution) rather than to error.

Corner cases
============

* ``R == b``: differential is ``0``, template left untouched.
* ``R < b``: differential clamped to ``0`` (no de-convolution).
* Different presmoothing for I / Q / U (e.g. ``56 arcmin`` / ``53 arcmin`` /
  ``53 arcmin``): each component is smoothed by its own differential, so all
  three land exactly at :math:`R`.
* A header value that cannot be parsed (or a missing keyword) is silently
  treated as ``0``.

MPI / distributed smoothing
===========================

The distributed (`libsharp`) smoothing path of
:func:`pysm3.apply_smoothing_and_coord_transform` supports the scalar ``fwhm``
path only; the per-component ``beam_window`` path is serial. At present the
build-time ``smoothing_angle`` differential smoothing is applied on the serial
path.

Tests
=====

The full feature is exercised in ``tests/test_presmoothing.py``: header round
trips, the differential-FWHM and window-division helpers (including the
``R <= pre`` identity), per-component presmoothing, end-to-end validation that
the output lands exactly at the target (and is far closer to a single target
smoothing than the naive double-smoothing), the resolution-floor no-op, Sky()
forwarding, and backwards compatibility.
