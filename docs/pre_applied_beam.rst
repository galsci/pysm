Pre-applied beam and differential smoothing
===========================================

Most PySM templates are beam-free by construction: small-scale
fluctuations are added on top of the observed large scales, so a
requested beam is applied in full.

A component whose templates are **not** beam-free, for example because
they are built from a survey map kept at its native resolution (e.g. the
Haslam 408 MHz map at 56 arcmin for the low-frequency synchrotron model)
or pre-simulated and smoothed to a fiducial beam (e.g. ``rg3``), can
declare the beam its templates already carry with the
``pre_applied_fwhm`` keyword, available on the template models that
support it (:py:class:`~pysm3.PowerLaw` and
:py:class:`~pysm3.CurvedPowerLaw`):

.. code-block:: toml

    [my_component]
    class = "CurvedPowerLaw"
    map_I = "my_template_1deg.fits"
    freq_ref_I = "408 MHz"
    map_pl_index = "my_beta_1deg.fits"
    pre_applied_fwhm = "1 deg"

The value is any angular quantity string parseable by ``astropy.units``.
The templates should generally be built to a common resolution once and
forever at data-preparation time, differentially smoothing the input
maps from their native beam to the target resolution with
:py:func:`~pysm3.get_differential_fwhm`:

.. code-block:: python

    import astropy.units as u
    import pysm3

    diff = pysm3.get_differential_fwhm(1 * u.deg, 56 * u.arcmin)  # ~21.5 arcmin
    template_1deg = pysm3.apply_smoothing_and_coord_transform(
        template_56arcmin, fwhm=diff
    )

The beam declared with ``pre_applied_fwhm`` is attached to the maps
returned by ``get_emission`` of the model and of the
:py:class:`~pysm3.Sky`, and :py:func:`~pysm3.apply_smoothing_and_coord_transform`
detects it: when a target ``fwhm`` is requested, only the **differential**
beam ``sqrt(max(fwhm**2 - pre_applied_fwhm**2, 0))`` is applied, in
quadrature, so the requested resolution is not over-smoothed by
double-applying part of the beam the map already has:

.. code-block:: python

    sky = pysm3.Sky(nside=256, preset_strings=["my_component"])
    m = sky.get_emission(0.408 * u.GHz)
    m.pre_applied_fwhm            # <Quantity 1. deg>
    m_smooth = pysm3.apply_smoothing_and_coord_transform(m, fwhm=2 * u.deg)
    # applies only sqrt(2**2 - 1**2) deg

A target ``fwhm`` not larger than the pre-applied beam results in no
smoothing, with a warning, since a map cannot be deconvolved. For maps
that were processed after ``get_emission`` (arithmetic, slicing and unit
conversions all drop the attached beam), pass the pre-applied beam
explicitly:

.. code-block:: python

    m_smooth = pysm3.apply_smoothing_and_coord_transform(
        m, fwhm=1.5 * u.deg, pre_applied_fwhm=1 * u.deg
    )

A custom ``beam_window`` is always applied in full and the pre-applied
beam is ignored, with a warning.

All the components of a multi-component :py:class:`~pysm3.Sky` must
share the same ``pre_applied_fwhm`` (or all be beam-free): the summed
emission of components at different resolutions would not have a single
well-defined beam and any smoothing would be wrong for at least one of
them, so a :py:class:`~pysm3.Sky` mixing them, for example
``Sky(preset_strings=["s8", "d9"])``, raises a ``ValueError`` at
construction. Combine components delivered at the same resolution, or
create and smooth separate :py:class:`~pysm3.Sky` objects.
