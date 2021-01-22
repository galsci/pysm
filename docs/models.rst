.. _models:

Summary of Models
*****************

For all details of the models also check the `presets.cfg file <https://github.com/healpy/pysm/blob/master/pysm3/data/presets.cfg>`_.

Input templates
===============

The PySM input templates are not stored in the Python package, they are downloaded on the fly when requested using the machinery from the `astropy.io.data` package.
The templates are stored in the `pysm-data <https://github.com/healpy/pysm-data>`_ repository on Github and published `via the cmb organization at NERSC <https://portal.nersc.gov/project/cmb/pysm-data/>`_.

PySM has an hardcoded path to access the files at NERSC, therefore when running on NERSC supercomputers or on Jupyter@NERSC, files are accessed directly.

When PySM is executed in another environment, the necessary templates are downloaded and cached in the home folder by astropy, see `the astropy.utils.data documentation <https://docs.astropy.org/en/stable/utils/data.html>`_ for the available configuration options.

Another option is to download the whole data repository, currently ~700MB (it will be way larger in the future), and define an environment variable::

    git clone https://github.com/healpy/pysm-data /data/pysm-data
    export PYSM_LOCAL_DATA=/data/pysm-data

Reproduce PySM 2 template preprocessing
=======================================

The PySM 2 paper describes how input data (e.g. component separation maps from Planck) have been processed, for example
most templates have been smoothed to remove high $\ell$ noise and then added small scale fluctuations.
Here we try to reproduce the process to clarify it:

* `Dust polarization templates from Planck Commander component separation outputs <preprocess-templates/reproduce_pysm2_dust_pol.html>`_, used in all dust models from `d0` to `d8` except `d4`
* `Synchrotron polarization templates from WMAP low frequency maps <preprocess-templates/reproduce_pysm2_sync_pol.html>`_, used in all synchrotron models


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

Synchrotron
===========

- **s1**: A power law scaling is used for the synchrotron emission, with a spatially varying spectral index. The emission templates are the Haslam 408 MHz, 57' resolution data reprocessed by Remazeilles et al 2015 MNRAS 451, 4311, and the WMAP 9-year 23 GHz Q/U maps (Bennett, C.L., et.al., 2014, ApJS, 208, 20B). The polarization maps have been smoothed with a Gaussian kernel of FWHM 5 degrees and had small scales added. The intensity template has had small scales added straight to the template. The details of the small scale procedure is outlined in the accompanying paper. The spectral index map was derived using a combination of the Haslam 408 MHz data and WMAP 23 GHz 7-year data (Miville-Deschenes, M.-A. et al., 2008, A&A, 490, 1093). The same scaling is used for intensity and polarization. This is the same prescription as used in the Planck Sky Model's v1.7.8 'power law' option (Delabrouille et al. A&A 553, A96, 2013), but with the Haslam map updated to the Remazeilles version. A 'curved power law' model is also supported with a single isotropic curvature index. The amplitude of this curvature is taken from Kogut, A. 2012, ApJ, 753, 110.

- **s2**: synchrotron index steepens off the Galactic plane, from -3.0 in the plane to -3.3 off the plane. Consistent with WMAP.

- **s3**: a power law with a curved index. The model uses the same index map as the nominal model, plus a curvature term. We use the best-fit curvature amplitude of -0.052 found in Kogut, A. 2012, ApJ, 753, 110, pivoted at 23 GHz.


AME
===

- **a1**: We model the AME as a sum of two spinning dust populations based on the Commander code (Planck Collaboration 2015, arXiv:1502.01588). A component is defined by a degree-scale emission template at a reference frequency and a peak frequency of the emission law. Both populations have a spatially varying emission template, one population has a spatially varying peak frequency, and the other population has a spatially constant peak frequency. The emission law is generated using the SpDust2 code (Ali-Haimoud 2008). The nominal model is unpolarized. We add small scales to the emission maps, the method is outlined in the accompanying paper.
  
- **a2**: AME has 2% polarization fraction. Polarized maps simulated with thermal dust angles and nominal AME intensity scaled globally by polarization fraction. Within WMAP/Planck bounds.


Free-free
=========

- **f1**: We model the free-free emission using the analytic model assumed in the Commander fit to the Planck 2015 data (Draine 2011 'Physics of the Interstellar and Intergalactic Medium') to produce a degree-scale map of free-free emission at 30 GHz. We add small scales to this using a procedure outlined in the accompanying paper. This map is then scaled in frequency by applying a spatially constant power law index of -2.14.

CMB
===

- **c1**: A lensed CMB realisation is computed using Taylens, a code to compute a lensed CMB realisation using nearest-neighbour Taylor interpolation (`taylens <https://github.com/amaurea/taylens>`_; Naess, S. K. and Louis, T. JCAP 09 001, 2013, astro-ph/1307.0719). This code takes, as an input, a set of unlensed Cl's generated using `CAMB <http://www.camb.info/>`_. The params.ini is in the Ancillary directory. There is a pre-computed CMB map provided at Nside 512.

