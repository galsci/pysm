Usage
=====

Generate a full-sky map of the Galactic foregrounds::

    import pysm3
    import pysm3.units as u

    sky = pysm3.Sky(nside=64, preset_strings=['d1', 's1', 'a1', 'f1'])
    m = sky.get_emission(30 * u.GHz)

The frequency must be an `astropy.units.Quantity` convertible to GHz (`30.0` without units raises an error).

`get_emission` returns the I, Q, U maps as an astropy `Quantity` of shape ``(3, npix)`` in RING ordering, in ``uK_RJ`` by default. Convert to ``uK_CMB`` with::

    m = m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(30 * u.GHz))

Each component of the sky is a subclass of :py:class:`~pysm3.Model`; check the :ref:`models` page for the full list of presets and the `pysm3 API reference <api/pysm3>`_ for the documentation of the classes.
