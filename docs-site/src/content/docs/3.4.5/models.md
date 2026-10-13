---
title: Models
description: "Summary of all built-in PySM models: dust, synchrotron, AME,
  free-free, CMB, CO lines, CIB, SZ emission and radio galaxies."
slug: 3.4.5/models
---

PySM provides a set of pre-configured models for each foreground component. Each `pysm3.Model` instance also knows about the maximum resolution available via the `max_nside` attribute. For all details check the `presets.cfg` file in the [repository](https://github.com/galsci/pysm/blob/main/pysm3/data/presets.cfg).

## Input templates

The PySM input templates are not stored in the Python package, they are downloaded on the fly when requested using the machinery from the `astropy.io.data` package. The templates are stored in the [pysm-data](https://github.com/galsci/pysm-data) repository on GitHub and published via the [cmb organization at NERSC](https://portal.nersc.gov/project/cmb/pysm-data/).

PySM has a hardcoded path to access the files at NERSC, therefore when running on NERSC supercomputers or on Jupyter@NERSC, files are accessed directly.

When PySM is executed in another environment, the necessary templates are downloaded and cached in the home folder by astropy, see the [astropy.utils.data documentation](https://docs.astropy.org/en/stable/utils/data.html) for the available configuration options.

Another option is to download the whole data repository, currently ~700MB (it will be way larger in the future), and define an environment variable:

```bash
git clone https://github.com/galsci/pysm-data /data/pysm-data
export PYSM_LOCAL_DATA=/data/pysm-data
```

## Dust

* **d1**: Thermal dust is modelled as a single-component modified black body (mbb). We use dust templates for emission at 545 GHz in intensity and 353 GHz in polarisation from the Planck-2015 analysis, and scale these to different frequencies with a mbb spectrum using the spatially varying temperature and spectral index obtained from the Planck data using the Commander code (Planck Collaboration 2015, arXiv:1502.01588). Note that it therefore assumes the same spectral index for polarization as for intensity. The input intensity template at 545 GHz is simply the available 2048 product degraded to nside 512. The polarization templates have been smoothed with a Gaussian kernel of FWHM 2.6 degrees, and had small scales added via the procedure described in the accompanying paper.

* **d0**: Simplified version of the **d1** model with a fixed spectral index of 1.54 and a black body temperature of 20 K.

* **d2** (**d3**): emissivity that varies spatially on degree scales, drawn from a Gaussian with $\beta=1.59 \pm 0.2$ (0.3). A Gaussian variation is not physically motivated, but amount of variation consistent with Planck.

* **d4**: a generalization of model 1 to multiple dust populations. It has been found that a two component model is still a good fit to the Planck data. This option uses the two component model from Finkbeiner, D. P., Davis, M., & Schlegel, D. J. 1999, Astrophysical Journal, 524, 867.

* **d5**: implementation of the dust model described in Hensley and Draine 2017.

* **d6**: implementation of the frequency decorrelation of dust, modelling the impact of averaging over spatially varying dust spectral indices both unresolved and along the line of sight. We take an analytic frequency covariance (Vansyngel 2016 arXiv:1611.02577) to calculate the resulting frequency dependence. The user specifies a single parameter, the correlation length. The smaller the correlation length, the larger the decorrelation. This parameter is constant across the sky.

* **d7**: modification of **d5** with iron inclusions in the grain composition.

* **d8**: simplified version of **d7** where the interstellar radiation field (ISRF) strength, instead of being a random realization, is fixed at 0.2. This corresponds reasonably well to a Modified Black Body model with temperature of 20K and an index of 1.54.

* **d9**: simplified version of **d10** with a fixed spectral index of 1.48 and a fixed dust black body temperature of 19.6 K all over the sky, based on Planck 2018 results.

* **d10**: single component modified black body model based on templates from the [GNILC needlet-based analysis of Planck data](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Foreground_maps#GNILC_thermal_dust_maps), with reduced contamination from CIB and point sources compared to the Commander maps used on **d1**. Small-scale fluctuations up to an $\ell\_{max}$ of 16384 have been added to the templates in the `logpoltens` (Logarithm of the Polarization Fraction Tensor) formalism. Also the spectral index and dust temperature maps have random fluctuations at small scales. Available up to $N\_{side}$ of 8192. Input templates are available [at NERSC](https://portal.nersc.gov/project/cmb/pysm-data/dust_gnilc/). Input templates are available at 2048, 4096 and 8192; lower nside are simulated by running `hp.ud_grade` on the $N\_{side}=2048$ maps. We also [apply a color correction](https://portal.nersc.gov/project/cmb/pysm-data/dust_gnilc/) to the 353 GHz template using the value provided by the [Planck 2018 XI paper](https://www.aanda.org/articles/aa/pdf/2020/09/aa32618-18.pdf) (Table 2). The galactic region of about 3% of the sky within the GAL 97 Planck mask is set to the input GNILC map at 21.8' to avoid excess power in the spectra caused by the injected small scale signal.

* **d11**: like **d10** with stochastic small scales generated on-the-fly. It can reproduce **d10** if configured to run with a specific set of seeds and a specific $\ell\_{max}$, see `pysm3.ModifiedBlackBodyRealization`. However reproducing **d10** is expensive because it needs to generate small scales with $\ell\_{max}=16384$ whatever output resolution is required, normally instead **d11** generates small scales just up to $\ell\_{max}=2.5N\_{side}$.

* **d12**: 3D model of polarized dust emission with 6 layers, based on the paper ["A 3-D model of polarised dust emission in the Milky Way"](https://arxiv.org/abs/1706.04162), named MKD based on the names of the authors. Each layer has different templates, spectral index and dust temperature. All maps were generated at Nside 2048 with the Planck Sky Model (PSM) by Jacques Delabrouille.

## Synchrotron

* **s1**: A power law scaling is used for the synchrotron emission, with a spatially varying spectral index. The emission templates are the Haslam 408 MHz, 57' resolution data reprocessed by Remazeilles et al 2015 MNRAS 451, 4311, and the WMAP 9-year 23 GHz Q/U maps (Bennett, C.L., et.al., 2013, ApJS, 208, 20B). The polarization maps have been smoothed with a Gaussian kernel of FWHM 5 degrees and had small scales added. The intensity template has had small scales added straight to the template. The spectral index map was derived using a combination of the Haslam 408 MHz data and WMAP 23 GHz 7-year data (Miville-Deschenes, M.-A. et al., 2008, A\&A, 490, 1093). The same scaling is used for intensity and polarization. This is the same prescription as used in the Planck Sky Model's v1.7.8 'power law' option (Delabrouille et al. A\&A 553, A96, 2013), but with the Haslam map updated to the Remazeilles version. A 'curved power law' model is also supported with a single isotropic curvature index. The amplitude of this curvature is taken from Kogut, A. 2012, ApJ, 753, 110.

* **s2**: synchrotron index steepens off the Galactic plane, from -3.0 in the plane to -3.3 off the plane. Consistent with WMAP.

* **s3**: a power law with a curved index. The model uses the same index map as the nominal model, plus a curvature term. We use the best-fit curvature amplitude of -0.052 found in Kogut, A. 2012, ApJ, 753, 110, pivoted at 23 GHz.

* **s4**: simplified version of **s5** with a fixed spectral index of -3.1 all over the sky.

* **s5**: power law model based on the same templates of **s1**, Haslam in temperature and WMAP 9 year 23 GHz in polarization. Small-scale fluctuations up to an $\ell\_{max}$ of 16384 have been added to the templates in the `logpoltens` (Logarithm of the Polarization Fraction Tensor) formalism. The spectral index map from **s1** has been rescaled [based on S-PASS](https://arxiv.org/abs/1802.01145) and had small scales added to upgrade it up to $N\_{side}$ 8192. Input templates are available [at NERSC](https://portal.nersc.gov/project/cmb/pysm-data/synch/) at 2048, 4096 and 8192; lower nside are simulated by running `hp.ud_grade` on the $N\_{side}=2048$ maps. The model replaces the 7% brightest part of the sky with the input maps to avoid having small scales with anomalous power due to the modulation which is proportional to the amplitude of the signal.

* **s6**: like **s5** with stochastic small scales generated on-the-fly. It can reproduce **s5** if configured to run with a specific set of seeds and a specific $\ell\_{max}$, see `pysm3.PowerLawRealization`. However reproducing **s5** is expensive because it needs to generate small scales with $\ell\_{max}=16384$ whatever output resolution is required, normally instead **s6** generates small scales just up to $\ell\_{max}=3N\_{side}-1$.

* **s7**: a power law with a curved index. The model uses the same templates and the same spectral index map of **s5**, the curvature term is based on the smoothed intensity template matched to the patch measured by the ARCADE experiment (Kogut, A. 2012, ApJ, 753, 110) and has random small scale fluctuations added. The curvature map is available at $N\_{side}=2048/4096/8192$.

## AME

* **a1**: We model the AME as a sum of two spinning dust populations based on the Commander code (Planck Collaboration 2015, arXiv:1502.01588). A component is defined by a degree-scale emission template at a reference frequency and a peak frequency of the emission law. Both populations have a spatially varying emission template, one population has a spatially varying peak frequency, and the other population has a spatially constant peak frequency. The emission law is generated using the SpDust2 code (Ali-Haimoud 2008). The nominal model is unpolarized. We add small scales to the emission maps, the method is outlined in [Thorne et al](https://arxiv.org/abs/1608.02841). This template comes from PySM 2, so the native resolution is $N\_{side}=512$.

* **a2**: AME has 2% polarization fraction. Polarized maps simulated with thermal dust angles and nominal AME intensity scaled globally by polarization fraction. Within WMAP/Planck bounds. This template comes from PySM 2, so the native resolution is $N\_{side}=512$.

## Free-free

* **f1**: We model the free-free emission using the analytic model assumed in the Commander fit to the Planck 2015 data (Draine 2011 'Physics of the Interstellar and Intergalactic Medium') to produce a degree-scale map of free-free emission at 30 GHz. We add small scales to this using a procedure outlined in [Thorne et al](https://arxiv.org/abs/1608.02841). This template comes from PySM 2, so the native resolution is $N\_{side}=512$. This map is then scaled in frequency by applying a spatially constant power law index of -2.14.

## CMB

* **c1**: A lensed CMB realisation is computed using Taylens, a code to compute a lensed CMB realisation using nearest-neighbour Taylor interpolation ([taylens](https://github.com/amaurea/taylens); Naess, S. K. and Louis, T. JCAP 09 001, 2013, astro-ph/1307.0719). This code takes, as an input, a set of unlensed Cl's generated using [CAMB](http://www.camb.info/). The unlensed Cl's were computed using CAMB with the best-fit ΛCDM cosmological parameters from the Planck 2015 TT,TE,EE+lowP data combination, as stated in Section 2.5 of the [PySM 2 paper](https://arxiv.org/abs/1608.02841) (Thorne et al. 2017, MNRAS 469, 2821) and detailed in [Planck 2015 results XIII. Cosmological parameters](https://arxiv.org/abs/1502.01589) (Planck Collaboration 2016, A\&A 594, A13, Table 4).

  The CAMB `params.ini` file used to generate the unlensed Cl's is included in the PySM 3 documentation: [`cmb_c1_params.ini`](https://github.com/galsci/pysm/blob/main/docs/cmb_c1_params.ini).

  The key cosmological parameters from `params.ini` are:

  | Parameter | Value | Description |
  | --- | --- | --- |
  | `ombh2` | 0.02225 | Physical baryon density, $\Omega_b h^2$ |
  | `omch2` | 0.1198 | Physical cold dark matter density, $\Omega_c h^2$ |
  | `omnuh2` | 0.00064 | Physical neutrino density, $\Omega\_\nu h^2$ |
  | `omk` | 0 | Curvature density, $\Omega_k$ (flat universe) |
  | `hubble` | 70 | Hubble constant, $H_0$ (km/s/Mpc) |
  | `temp_cmb` | 2.7255 | CMB temperature, $T\_{CMB}$ (K) |
  | `helium_fraction` | 0.24 | Primordial helium mass fraction, $Y\_{He}$ |
  | `massless_neutrinos` | 2.046 | Effective number of massless neutrinos, $N\_{eff}$ |
  | `re_optical_depth` | 0.06 | Reionization optical depth, $\tau$ |
  | `scalar_spectral_index(1)` | 0.9645 | Scalar spectral index, $n_s$ |
  | `scalar_amp(1)` | 2.1e-9 | Scalar amplitude, $A_s$ |
  | `pivot_scalar` | 0.05 | Scalar pivot scale, $k\_{pivot}$ (Mpc⁻¹) |
  | `initial_ratio(1)` | 0.0 | Tensor-to-scalar ratio, $r$ |
  | `l_max_scalar` | 2200 | Maximum multipole for scalar spectra |

  The parameter values match the Planck 2015 TT,TE,EE+lowP best-fit ΛCDM cosmology from Table 4 of [Planck 2015 results XIII](https://arxiv.org/abs/1502.01589) (Planck Collaboration 2016, A\&A 594, A13). The primary cosmological parameters ($\Omega_b h^2$, $\Omega_c h^2$, $n_s$, $T\_{CMB}$) match the Planck 2015 values exactly. Some secondary parameters differ slightly:

  | Parameter | Planck 2015 value | params.ini value |
  | --- | --- | --- |
  | $\Omega_b h^2$ | 0.02225 ± 0.00016 | `ombh2 = 0.02225` (exact match) |
  | $\Omega_c h^2$ | 0.1198 ± 0.0015 | `omch2 = 0.1198` (exact match) |
  | $n_s$ | 0.9645 ± 0.0049 | `scalar_spectral_index(1) = 0.9645` (exact match) |
  | $H_0$ | 67.27 ± 0.66 | `hubble = 70` (rounded) |
  | $\tau$ | 0.079 ± 0.017 | `re_optical_depth = 0.06` (lower) |
  | $A_s$ | $2.207e-9$ | `scalar_amp(1) = 2.1e-9` (≈5% lower) |

* **c2**: Precomputed lensed CMB map of the **c1** model at $N\_{side}=512$.

* **c3**: Unlensed CMB map with the same cosmological parameters of WebSky 0.4, available at $N\_{side}=512$ and $N\_{side}=4096$. Maps are generated with $\ell\_{max}=8250$.

* **c4**: CMB map with the same cosmological parameters and lensed with the convergence map of WebSky 0.4, available at $N\_{side}=512$ and $N\_{side}=4096$. Maps are generated with $\ell\_{max}=8250$.

## CMB Dipole

The dipole component models the kinematic dipole induced by the Earth's motion relative to the CMB rest frame (Solar Dipole). This is implemented in the `pysm3.models.dipole.CMBDipole` class.

* **dip1**: This model implements the Planck Public Release 4 Solar Dipole. The parameters for the amplitude and direction are taken from [Planck intermediate results. LVII. Joint Planck LFI and HFI data processing (2020)](https://doi.org/10.1051/0004-6361/202038073) and the CMB temperature from [The Temperature of the Cosmic Microwave Background (2009)](https://doi.org/10.1088/0004-637X/707/2/916). The dipole amplitude agrees with the model implemented in the [toast](https://toast-cmb.readthedocs.io/) package at the 1e-5 level.

* **dip2**: This model is the same as **dip1** with an additional frequency-dependent quadrupole correction. This correction accounts for the relativistic aberration and Doppler effects beyond the simple dipole, which induce a quadrupole moment in the observed CMB. For more details, see [On the proper kinetic quadrupole CMB removal and the quadrupole anomalies (2015)](https://doi.org/10.1088/1475-7516/2015/06/047). The quadrupole correction amplitude agrees with [toast](https://toast-cmb.readthedocs.io/) at the 1e-5 level.

## CO line emission

* **co1**: Galactic CO emission involving the first 3 CO rotational lines, i.e. $J=1-0,2-1,3-2$ whose center frequency is respectively at $\nu_0 = 115.3, 230.5, 345.8$ GHz. The CO emission map templates are the CO Planck maps obtained with `MILCA` component separation algorithm (See [Planck paper](https://www.aanda.org/articles/aa/abs/2014/11/aa21553-13/aa21553-13.html)). The CO maps have been released at the nominal resolution (10 and 5 arcminutes). However, to reduce noise contamination from template maps (especially at intermediate and high Galactic latitudes), we convolved them with a 1 deg gaussian beam.

* **co2**: like **co1** with polarized emission at the level of 0.1%.

* **co3**: like **co2** with a mock CO clouds map 20 degrees off the Galactic plane simulated with `MCMole3D`.

## Cosmic Infrared Background

* **cib1**: Cosmic Infrared Background map built from the WebSky 0.4 simulation, it is generated at several input frequencies and then linearly interpolated in `uK_RJ` by PySM. Available at $N\_{side}=4096$. It also includes an analytical correction to its amplitude to match the South Pole Telescope data.

## Sunyaev–Zeldovich emission

* **tsz1**: Thermal SZ emission from WebSky 0.4. Available at $N\_{side}=8192$.

* **tsz2**: Thermal SZ emission from the [Agora simulations](https://doi.org/10.1093/mnras/stae1031) (based on BAHAMAS hydrodynamical simulations). Available at $N\_{side}=8192$.

* **tsz3**: Lensed Thermal SZ emission from the [Agora simulations](https://doi.org/10.1093/mnras/stae1031). Available at $N\_{side}=8192$.

* **tsz4**: Thermal SZ emission from the [HalfDome simulations](https://doi.org/10.48550/arXiv.2407.17462) 0.1 (generated using xgpaint with Battaglia16 profiles). Eleven realizations available at $N\_{side}=8192$ by overriding the `template_name` with seeds: 100 (default), 102, 104, 106, 108, 110, 112, 114, 116, 118, 120. For example: `halfdome/0.1/tsz/y_b16_halo_res1_s102.fits`.

* **ksz1**: Kinetic SZ emission from WebSky 0.4. Available at $N\_{side}=4096$.

* **ksz2**: Kinetic SZ emission from the [Agora simulations](https://doi.org/10.1093/mnras/stae1031) (based on BAHAMAS hydrodynamical simulations). Available at $N\_{side}=8192$.

* **ksz3**: Lensed Kinetic SZ emission from the [Agora simulations](https://doi.org/10.1093/mnras/stae1031). Available at $N\_{side}=8192$.

* **ksz5**: Lensed Kinetic SZ emission from the [FLAMINGO simulations](https://doi.org/10.1093/mnras/stag625) (using the integrated maps for the fiducial feedback model: L1_m9). Available at $N\_{side}=4096$.

* **ksz6**: Lensed Kinetic SZ emission from the [FLAMINGO simulations](https://doi.org/10.1093/mnras/stag625) (using the integrated maps for the strongest feedback model: fgas-8sigma). Available at $N\_{side}=4096$.

## Radio galaxies

* **rg1**: Emission from Radio Galaxies simulated with WebSky 0.4, sources are not polarized. Available at $N\_{side}=4096$ at the same input frequencies of CIB and then interpolated.

* **rg2**: Brightest Radio Galaxies from WebSky 0.4, flux above `1 mJy` at `100 GHz`, about 370k individual point sources simulated on-the-fly in pixel space from a catalog of sources to the desired resolution and beam using `pixell.sim_objects`. The sources are polarized with a polarization angle generated randomly with a fixed seed, so they are consistent across runs. It is complementary to **rg3**. Relies on `pysm3.PointSourceCatalog`. The catalog itself is available at [NERSC](https://portal.nersc.gov/cfs/cmb/pysm-data/websky/0.4/radio_catalog/), the format is easy to read with `xarray`, it contains the coordinates of the sources and the polynomial coefficients to generate the flux at the desired frequency.

* **rg3**: Background Radio Galaxies from WebSky 0.4, flux below `1 mJy` at `100 GHz`, about 280 million individual point sources pre-simulated at Nside 2048, 4096 and 8192 at a set of input frequencies and smoothed in pixel space to a fiducial beam. It interpolates across HEALPix maps at different frequencies for bandpass integration then applies the differential beam in spherical harmonics space. Sources are polarized, same technique used for **rg2**. Relies on `pysm3.InterpolatingComponent`.
