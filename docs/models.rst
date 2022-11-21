.. _models:

Summary of Models
*****************

For all details of the models also check the `presets.cfg file <https://github.com/galsci/pysm/blob/main/pysm3/data/presets.cfg>`_.

Each :py:class:`~pysm3.Model` instance also knows about the maximum resolution available in the `max_nside` attribute.

Input templates
===============

The PySM input templates are not stored in the Python package, they are downloaded on the fly when requested using the machinery from the `astropy.io.data` package.
The templates are stored in the `pysm-data <https://github.com/galsci/pysm-data>`_ repository on Github and published `via the cmb organization at NERSC <https://portal.nersc.gov/project/cmb/pysm-data/>`_.

PySM has an hardcoded path to access the files at NERSC, therefore when running on NERSC supercomputers or on Jupyter@NERSC, files are accessed directly.

When PySM is executed in another environment, the necessary templates are downloaded and cached in the home folder by astropy, see `the astropy.utils.data documentation <https://docs.astropy.org/en/stable/utils/data.html>`_ for the available configuration options.

Another option is to download the whole data repository, currently ~700MB (it will be way larger in the future), and define an environment variable::

    git clone https://github.com/galsci/pysm-data /data/pysm-data
    export PYSM_LOCAL_DATA=/data/pysm-data

Reproduce PySM 2 template preprocessing
=======================================

The PySM 2 paper describes how input data (e.g. component separation maps from Planck) have been processed, for example
most templates have been smoothed to remove high $\ell$ noise and then added small scale fluctuations.
Here we try to reproduce the process to clarify it:

* `Dust polarization templates from Planck Commander component separation outputs <preprocess-templates/reproduce_pysm2_dust_pol.ipynb>`_, used in all dust models from `d0` to `d8` except `d4`
* `Synchrotron polarization templates from WMAP low frequency maps <preprocess-templates/reproduce_pysm2_sync_pol.ipynb>`_, used in all synchrotron models


Dust
====

- **d1**: Thermal dust is modelled as a single-component modified black body (mbb). We use dust templates for emission at 545 GHz in intensity and 353 GHz in polarisation from the Planck-2015 analysis, and scale these to different frequencies with a mbb spectrum using the spatially varying temperature and spectral index obtained from the Planck data using the Commander code (Planck Collaboration 2015, arXiv:1502.01588). Note that it therefore assumes the same spectral index for polarization as for intensity. The input intensity template at 545 GHz is simply the available 2048 product degraded to nside 512. The polarization templates have been smoothed with a Gaussian kernel of FWHM 2.6 degrees, and had small scales added via the procedure described in the accompanying paper.

- **d0**: Simplified version of the **d1** model with a fixed spectral index of 1.54 and a black body temperature of 20 K.

- **d2** (**d3**): emissivity that varies spatially on degree scales, drawn from a Gaussian with beta=1.59 \pm 0.2 (0.3). A Gaussian variation is not physically motivated, but amount of variation consistent with Planck.

- **d4**: a generalization of model 1 to multiple dust populations. It has been found that a two component model is still a good fit to the Planck data. This option uses the two component model from Finkbeiner, D. P., Davis, M., & Schlegel, D. J. 1999, Astrophysical Journal, 524, 867.

- **d5**: implementation of the dust model described in Hensley and Draine 2017.
  
- **d6**: implementation of the frequency decorrelation of dust, modelling the impact of averaging over spatially varying dust spectral indices both unresolved and along the line of sight. We take an analytic frequency covariance (Vansyngel 2016 arXiv:1611.02577) to calculate the resulting frequency dependence. The user specifies a single parameter, the correlation length. The smaller the correlation length, the larger the decorrelation. This parameter is constant across the sky.

- **d7**: modification of `d5` with iron inclusions in the grain composition.

- **d8**: simplified version of `d7` where the interstellar radiation field (ISRF) strength, instead of being a random realization, is fixed at 0.2.  This corresponds reasonably well to a Modifield Black Body model with temperature of 20K and an index of 1.54.

- **d9**: simplified version of **d10** with a fixed spectral index of 1.48 and a fixed dust black body temperature of 19.6 K all over the sky, based on Planck 2018 results.

- **d10**: single component modified black body model based on templates from the `GNILC needlet-based analysis of Planck data <https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Foreground_maps#GNILC_thermal_dust_maps>`_, with reduced contamination from CIB and point sources compared to the Commander maps used on **d1**. Small-scale fluctuations up to a $\ell_{max}$ of 16384 have been added to the templates in the `logpoltens` (Logarithm of the Polarization Fraction Tensor) formalism. Also the spectral index and dust temperature maps have random fluctuations at small scales. Available up to $N_{side}$ of 8192. Input templates are available `at NERSC <https://portal.nersc.gov/project/cmb/pysm-data/dust_gnilc/>`_.  Input templates are available at 2048, 4096 and 8192; lower nside are simulated by running `hp.ud_grade` on the $N_{side}=2048$ maps.  For more details about the pre-processing of the input data, see the `notebook about creating the GNILC-based template and adding small scales using the logpoltens formalism <preprocess-templates/small_scale_dust_pysm3.ipynb>`_ and `the notebook about spectral index and dust temperature <preprocess-templates/gnilc_dust_spectralindex_Tdust.ipynb>`_. We also `apply a color correction <preprocess-templates/small_scale_dust_pysm3_generate_templates.ipynb>`_ to the 353 GHz template using the value provided by the `Planck 2018 XI paper <https://www.aanda.org/articles/aa/pdf/2020/09/aa32618-18.pdf>`_ (Table 2). The galactic region of about 3% of the sky within the GAL 97 Planck mask is set to the input GNILC map at 21.8' to avoid excess power in the spectra caused by the injected small scale signal.

- **d11**: like **d10** with stochastic small scales generated on-the-fly. It can reproduce **d10** if configured to run with a specific set of seeds and a specific $\ell_{max}$, see :py:class:`~pysm3.ModifiedBlackBodyRealization`. However reproducing **d10** is expensive because it needs to generate small scales with a $\ell_{max}=16384$ whatever output resolution is required, normally instead **d11** generates small scales just up to $\ell_{max}=2.5N_{side}$.

- **d12**: 3D model of polarized dust emission with 6 layers, based on the paper `"A 3-D model of polarised dust emission in the Milky Way" <https://arxiv.org/abs/1706.04162>`_, named MKD based on the names of the authors. Each layer has different templates, spectral index and dust temperature. All maps were generated at N_side 2048 with the Planck Sky Model (PSM) by Jacques Delabrouille.

Synchrotron
===========

- **s1**: A power law scaling is used for the synchrotron emission, with a spatially varying spectral index. The emission templates are the Haslam 408 MHz, 57' resolution data reprocessed by Remazeilles et al 2015 MNRAS 451, 4311, and the WMAP 9-year 23 GHz Q/U maps (Bennett, C.L., et.al., 2013, ApJS, 208, 20B). The polarization maps have been smoothed with a Gaussian kernel of FWHM 5 degrees and had small scales added. The intensity template has had small scales added straight to the template. The details of the small scale procedure is outlined in the accompanying paper. The spectral index map was derived using a combination of the Haslam 408 MHz data and WMAP 23 GHz 7-year data (Miville-Deschenes, M.-A. et al., 2008, A&A, 490, 1093). The same scaling is used for intensity and polarization. This is the same prescription as used in the Planck Sky Model's v1.7.8 'power law' option (Delabrouille et al. A&A 553, A96, 2013), but with the Haslam map updated to the Remazeilles version. A 'curved power law' model is also supported with a single isotropic curvature index. The amplitude of this curvature is taken from Kogut, A. 2012, ApJ, 753, 110.

- **s2**: synchrotron index steepens off the Galactic plane, from -3.0 in the plane to -3.3 off the plane. Consistent with WMAP.

- **s3**: a power law with a curved index. The model uses the same index map as the nominal model, plus a curvature term. We use the best-fit curvature amplitude of -0.052 found in Kogut, A. 2012, ApJ, 753, 110, pivoted at 23 GHz.

- **s4**: simplified version of **s5** with a fixed spectral index of -3.1 all over the sky.

- **s5**: power law model based on the same templates of **s1**, Haslam in temperature and WMAP 9 year 23 GHz in polarization. Small-scale fluctuations up to a $\ell_{max}$ of 16384 have been added to the templates in the `logpoltens` (Logarithm of the Polarization Fraction Tensor) formalism. The spectral index map from **s1** has been rescaled `based on S-PASS <https://arxiv.org/abs/1802.01145>`_ and had small scales added to upgrade it up to $N_{side}$ 8192. Input templates are available `at NERSC <https://portal.nersc.gov/project/cmb/pysm-data/synch/>`_ at 2048, 4096 and 8192; lower nside are simulated by running `hp.ud_grade` on the $N_{side}=2048$ maps.  For more details about the pre-processing of the input data, see the `notebook about creating the templates and adding small scales using the logpoltens formalism <preprocess-templates/synchrotron_template_logpoltens.ipynb>`_ and `the notebook about spectral index <preprocess-templates/synchrotron_beta.ipynb>`_.

- **s6**: like **s5** with stochastic small scales generated on-the-fly. It can reproduce **s5** if configured to run with a specific set of seeds and a specific $\ell_{max}$, see :py:class:`~pysm3.PowerLawRealization`. However reproducing **s5** is expensive because it needs to generate small scales with a $\ell_{max}=16384$ whatever output resolution is required, normally instead **s6** generates small scales just up to $\ell_{max}=3N_{side}-1$.

- **s7**: a power law with a curved index. The model uses the same templates and the same spectral index map of **s5**, the curvature term is based on the smoothed intensity template matched to the patch measured by the ARCADE experiment (Kogut, A. 2012, ApJ, 753, 110) and has random small scale fluctuations added, see `the relevant notebook <preprocess-templates/synchrotron_curvature.ipynb>`_. The curvature map is available at $N_{side}=2048/4096/8192$.


AME
===

- **a1**: We model the AME as a sum of two spinning dust populations based on the Commander code (Planck Collaboration 2015, arXiv:1502.01588). A component is defined by a degree-scale emission template at a reference frequency and a peak frequency of the emission law. Both populations have a spatially varying emission template, one population has a spatially varying peak frequency, and the other population has a spatially constant peak frequency. The emission law is generated using the SpDust2 code (Ali-Haimoud 2008). The nominal model is unpolarized. We add small scales to the emission maps, the method is outlined in the accompanying paper.
  
- **a2**: AME has 2% polarization fraction. Polarized maps simulated with thermal dust angles and nominal AME intensity scaled globally by polarization fraction. Within WMAP/Planck bounds.


Free-free
=========

- **f1**: We model the free-free emission using the analytic model assumed in the Commander fit to the Planck 2015 data (Draine 2011 'Physics of the Interstellar and Intergalactic Medium') to produce a degree-scale map of free-free emission at 30 GHz. We add small scales to this using a procedure outlined in the accompanying paper. This map is then scaled in frequency by applying a spatially constant power law index of -2.14.

CMB
===

- **c1**: A lensed CMB realisation is computed using Taylens, a code to compute a lensed CMB realisation using nearest-neighbour Taylor interpolation (`taylens <https://github.com/amaurea/taylens>`_; Naess, S. K. and Louis, T. JCAP 09 001, 2013, astro-ph/1307.0719). This code takes, as an input, a set of unlensed Cl's generated using `CAMB <http://www.camb.info/>`_. The params.ini is in the Ancillary directory.

- **c2**: Precomputed lensed CMB map of the **c1** model at $N_{side}=512$.

- **c3**: Unlensed CMB map with the same cosmological parameters of WebSky 0.4, available at $N_{side}=512$ and $N_{side}=4096$. Maps are generated with $\ell_{max}=8250$. For more details see :ref:`websky`.

- **c4**: CMB map with the same cosmological parameters and lensed with the convergence map of WebSky 0.4 , available at $N_{side}=512$ and $N_{side}=4096$. Maps are generated with $\ell_{max}=8250$.

CO line emission
================

.. toctree::
  :maxdepth: 2

  co_lines

- **co1**: Galactic CO emission involving the first 3 CO rotational lines, i.e. :math:`J=1-0,2-1,3-2` whose center frequency is respectively at :math:`\nu_0 = 115.3, 230.5,345.8` GHz. The CO emission map templates are the CO Planck maps obtained with ``MILCA`` component separation algorithm (See `Planck paper <https://www.aanda.org/articles/aa/abs/2014/11/aa21553-13/aa21553-13.html>`). The CO maps have been released at the nominal resolution (10 and 5 arcminutes). However, to reduce  noise contamination from template maps (especially at intermediate and high Galactic latitudes), we  convolved them with a 1 deg gaussian beam.
- **co2**: like **co1** with polarized emission at the level of 0.1%.
- **co3**: like **co2** with a mock CO clouds map 20 degrees off the Galactic plane simulated with ``MCMole3D``.

Cosmic Infrared Background
==========================

- **cib1**: Cosmic Infrared Background map built from the WebSky 0.4 simulation, it is generated at several input frequencies and then linearly interpolated in `uK_RJ` by PySM. Available at $N_{side}=4096$. It also includes an analytical correction to its amplitude to match the South Pole Telescope data. For more details see :ref:`websky`.

Sunyaev–Zeldovich emission
==========================

- **tsz1**: Thermal SZ emission from WebSky 0.4. Available at $N_{side}=4096$. For more details see :ref:`websky`.

- **ksz1**: Kinetic SZ emission from WebSky 0.4. Available at $N_{side}=4096$. For more details see :ref:`websky`.

Radio galaxies
==============

- **rg1**: Emission from Radio Galaxies simulated with WebSky 0.4. Available at $N_{side}=4096$ at the same input frequencies of CIB and then interpolated. For more details see :ref:`websky`.
