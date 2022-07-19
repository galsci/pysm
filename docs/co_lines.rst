.. _colines:

COLines
=======

This class implements simulations for Galactic CO emission involving the first 3 CO rotational lines, i.e. :math:`J=1-0,2-1,3-2` whose center frequency is respectively at :math:`\nu_0 = 115.3, 230.5,345.8` GHz. The CO emission map templates are the CO Planck maps obtained with ``MILCA`` component separation algorithm (See `Planck paper <https://www.aanda.org/articles/aa/abs/2014/11/aa21553-13/aa21553-13.html>`). The CO maps have been released at the nominal resolution (10 and 5 arcminutes). However, to reduce  noise contamination from template maps (especially at intermediate and high Galactic latitudes), we  convolved them with a 1 deg gaussian beam.

The Stokes I map is computed from the template one as it follows:

if target :math:`N_{side}` <= 512:

    #. The template map at a ``nside=512``  is downgraded at the target :math:`N_{side}`

if target :math:`N_{side}` > 512 :

    #. The template map at a ``nside=2048``  is downgraded(eventually upgraded) at the target :math:`N_{side}`

Q and U maps can be computed from the template CO emission  map, :math:`I_{CO}`,  assuming a constant  fractional polarization, as:

.. math::

    Q = f_{pol} I_{CO}  g_d \cos( 2 \psi)

    U  = f_{pol} I_{CO}  g_d \sin( 2 \psi)

with :math:`g_d` and :math:`\psi` being respectively the depolarization and polarization angle maps estimated from a dust map as :

.. math::

    g_d = \frac{ \sqrt{Q^2_{d,353}    + U^2_{d,353}   } }{f_{pol} I_{d,353} }

    \psi = \frac{1}{2} \arctan {\frac{U_{d,353}}{Q_{d,353}}}


Most of the CO emission is expected to be confined in the  Galactic midplane. However, there are still regions at high Galactic latitudes  where the CO emission has been purely assessed (by current surveys) and where the Planck signal-to-noise was not enough to detect any emission.

The PySM user can include the eventuality of molecular emission (both unpolarized and polarized) at High Gal. Latitudes by co-adding to the emission maps one realization of CO emission simulated with MCMole3D together with  the Planck CO map. The polarization is simulated similarly as above.

The ``MCMole3D`` input parameters  are are obtained from best fit with the Planck CO 1-0 map (see Puglisi et al. 2017 and the `documentation <http://giuspugl.github.io/mcmole/index.html>`_). If ``include_high_galactic_latitude_clouds=True``, a mock CO cloud map is simulated with ``MCMole3D``, encoding high Galactic latitudes clouds at latitudes above and below  than 20 degrees. The mock emission map is then co-added to the Planck CO emission map. The polarization is simulated similarly as above.

The installation of ``mcmole3d`` is not required, HGL clouds can be input to the CO emission by setting ``run_mcmole3d=False``  (which is the default). However, if one wants to run several mock CO  realizations observing high Galactic latitude patches we encourage to run ``mcmole3d`` by changing ``random_seed`` in the CO class constructor. The parameter ``theta_high_galactic_latitude_deg`` set the latitude above which CO emission from high Galactic latitudes can be included and it has an impact **only when** ``run_mcmole3d=True``.

The level of polarization in **co2** and **co3** is 0.1%, which on average is the expected level on 10% of the sky. However, polarization from CO emission have been detected at larger fluxes in  Orion and Taurus complexes (Greaves et al.1999 )

See `this post <https://giuspugl.github.io/reports/Adding_CO_to_pysm>`_ for the actual processing of templates.
