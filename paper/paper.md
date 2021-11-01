---
title: 'The Python Sky Model 3 software'
tags:
  - cosmology
  - astronomy
  - python
authors:
 - name: Andrea Zonca
   orcid: 0000-0001-6841-1058
   affiliation: "1"
 - name: Ben Thorne
   orcid: 0000-0002-0457-0153
   affiliation: "2"
 - name: Nicoletta Krachmalnicoff
   affiliation: "3,4,5"
 - name: Julian Borrill
   affiliation: "6,7"
affiliations:
 - name: San Diego Supercomputer Center, University of California San Diego, San Diego, USA
   index: 1
 - name: Department of Physics, University of California Davis, One Shields Avenue, Davis, CA 95616, USA
   index: 2
 - name: SISSA, Via Bonomea 265, 34136 Trieste, Italy
   index: 3
 - name: INFN, Via Valerio 2, 34127 Trieste, Italy
   index: 4
 - name: IFPU, Via Beirut 2, 34014 Trieste, Italy
   index: 5
 - name: Computational Cosmology Center, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 6
 - name: Space Sciences Laboratory at University of California, 7 Gauss Way, Berkeley, CA 94720
   index: 7
date: 22 July 2021
bibliography: paper.bib
---

# Statement of Need

The Cosmic Microwave Background (CMB) radiation, emitted just 370 thousand years after the Big Bang, is a pristine probe of the Early Universe. After being emitted at high temperatures, the CMB was redshifted by the subsequent 13.8 billion years of cosmic expansion, such that it is brightest at microwave frequencies today.
However, our own Milky Way galaxy also emits in the microwave portion of the spectrum, obscuring our view of the CMB.  Examples of this emission are thermal radiation by interstellar dust grains and synchrotron emission by relativistic electrons spiraling in magnetic fields.
Cosmologists need to create synthetic maps of the CMB and of the galactic emission based on available data and on physical models that extrapolate observations to different frequencies. The resulting maps are useful to test data reduction algorithms, to understand residual systematics, to forecast maps produced by future instruments, to run Monte Carlo analysis for noise estimation, and more.

# Summary

The Python Sky Model (PySM) is a Python package used by Cosmic Microwave Background (CMB) experiments to simulate maps, in HEALPix [@gorski05; @healpy09] pixelization, of the various diffuse astrophysical components of Galactic emission relevant at CMB frequencies (i.e., dust, synchrotron, free-free and Anomalous Microwave Emission), as well as the CMB itself. These maps may be integrated over a given instrument bandpass and smoothed with a given instrument beam.
The template emission maps used by PySM are based on Planck [@planck18] and WMAP [@wmap13] data and are noise-dominated at small scales. Therefore, PySM simulation templates are smoothed to retain the large-scale information, and then supplemented with modulated Gaussian realizations at smaller scales. This strategy allows one to simulate data at higher resolution than the input maps.

PySM 2 [@pysm17], released in 2016, has become the de-facto standard for simulating Galactic emission; it is used, for example, by CMB-S4, Simons Observatory, LiteBird, PICO, CLASS, POLARBEAR, and other CMB experiments, as shown by the [80+ citations of the PySM 2 publication](https://scholar.google.com/scholar?start=0&hl=en&as_sdt=2005&sciodt=0,5&cites=16628417670342266167&scipsc=).
As the resolution of upcoming experiments increases, the PySM 2 software has started to show some limitations:

* Emission templates are provided at 7.9 arcminutes resolution (HEALPix $N_{side}=512$), while the next generation of CMB experiments will require sub-arcminute resolution.
* The software is implemented in pure `numpy`, meaning that it has significant memory overhead and is not multi-threaded, precluding simply replacing the current templates with higher-resolution versions.
* Emission templates are included in the PySM 2 Python package, which is still practical when each of the roughly 40 input maps is ~10 Megabytes, but will not be if they are over 1 Gigabyte.

The solution to these issues was to reimplement PySM from scratch focusing of these features:

* Reimplement all the models with the `numba` [@numba] Just-In-Time compiler for Python to reduce memory overhead and optimize performance: the whole integration loop of a template map over the frequency response of an instrument is performed in a single pass in automatically compiled and multi-threaded Python code.
* Use MPI through `mpi4py` to coordinate execution of PySM 3 across multiple nodes, this allows supporting template maps at a resolution up to 0.4 arcminutes (HEALPix $N_{side}=8192$).
* Rely on `libsharp` [@libsharp], a distributed implementation of spherical harmonic transforms, to smooth the maps with the instrument beam when maps are distributed over multiple nodes with MPI.
* Employ the data utilities infrastructure provided by `astropy` [@astropy2013; @astropy2018] to download the input templates and cache them when requested.

At this stage we strive to maintain full compatibility with PySM 2, therefore we implement the exact same astrophysical emission models with the same naming scheme. In the extensive test suite we compare the output of each PySM 3 model with the results obtained by PySM 2.

# Performance

As an example of the performance improvements achieved with PySM 3 over PySM 2, we run the following configuration:

* An instrument with 3 channels, with different beams, and a top-hat bandpass defined numerically at 10 frequency samples.
* A sky model with the simplest models of dust, synchrotron, free-free and AME [`a1,d1,s1,f1` in PySM terms].
* Execute on a 12-core Intel processor with 12 GB of RAM.

The following tables shows the walltime and peak memory usage of this simulation executed at the native PySM 2 resolution of $N_{side}=512$ and at two higher resolutions:

| Output $N_{side}$ | PySM 3        | PySM 2        |
|-------------------|---------------|---------------|
| 512               | 1m 0.7 GB     | 1m40s 1.45 GB |
| 1024              | 3m30s 2.3 GB  | 7m20s 5.5 GB  |
| 2048              | 16m10s 8.5 GB | Out of memory |

The models at $N_{side}=512$ have been tested to be equal given a relative tolerance of `1e-5`.

At the moment it is not very useful to run at resolutions higher than $N_{side}=512$ because there is no actual template signal at smaller scales. However, this demonstrates the performance improvements that will make working with higher resolution templates possible.

# Future work

PySM 3 opens the way to implement a new category of models at much higher resolution. However, instead of just upgrading the current models to smaller scales, we want to also update them with the latest knowledge of Galactic emission and gather feedback from each of the numerous CMB experiments. For this reason we are collaborating with the Panexperiment Galactic Science group to lead the development of the new class of models to be included in PySM 3.

# How to cite

If you are using PySM 3 for your work, please cite this paper for the software itself; for the actual emission modeling please also cite the original PySM 2 paper [@pysm17]. There will be a future paper on the generation of new PySM 3 astrophysical models.

# Acknowledgments

* This work was supported in part by NASA grant `80NSSC18K1487`.
* The software was tested, in part, on facilities run by the Scientific Computing Core of the Flatiron Institute.
* This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility located at Lawrence Berkeley National Laboratory, operated under Contract No. `DE-AC02-05CH11231`.

# References
