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
   affiliation: "3"
 - name: Julian D. Borrill
   affiliation: "4,5,6"
affiliations:
 - name: San Diego Supercomputer Center, University of California, San Diego, USA
   index: 1
 - name: Department of Physics, University of California, One Shields Avenue, Davis, CA 95616, USA
   index: 2
 - name: TODO
   index: 3
 - name: Computational Cosmology Center, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 4
 - name: Space Sciences Laboratory at University of California, 7 Gauss Way, Berkeley, CA 94720
   index: 5
 - name: Department of Physics, University of California, Berkeley, CA, USA 94720
   index: 6
date: 20 July 2021
bibliography: paper.bib
---

# Summary

The Python Sky Model (PySM) is a Python package used by Cosmic Microwave Background (CMB) experiments to simulate maps, in HEALPix [@gorski05; @healpy09] pixelization, of the various diffuse astrophysical components of Galactic emission relevant at CMB frequencies (i.e. dust, synchrotron, free-free and Anomalous Microwave Emission), as well as the CMB itself. These maps may be integrated over a given instrument bandpass and smoothed with a given instrument beam.
The template emission maps used by PySM are based on Planck [@planck18] and WMAP [@wmap13] data and are noise-dominated at small scales. Therefore, PySM simulation templates are smoothed to retain the large-scale information, and then supplemented with modulated Gaussian realizations at smaller scales. This strategy allows one to simulate data at higher resolution than the input maps.

PySM 2 [@pysm17], released in 2016, has become the de-facto standard for simulating Galactic emission, for example it is used by CMB-S4, Simons Observatory, LiteBird, PICO, CLASS, POLARBEAR and other CMB experiments, as shown by the [80+ citations of the PySM 2 publication](https://scholar.google.com/scholar?start=0&hl=en&as_sdt=2005&sciodt=0,5&cites=16628417670342266167&scipsc=).
As the resolution of upcoming experiments increases, the PySM 2 software has started to show some limitations:

* Emission templates are provided at 7.9 arcminutes resolution (HEALPix $N_{side}=512$), while the next generation of CMB experiments will require sub-arcminute resolution.
* Just replacing the input maps in PySM 2 with higher-resoltion versions would not be sufficient as the software is implemented in pure `numpy`, meaning it has significant memory overhead and it is not multi-threaded. 
* Emission templates are included in the PySM 2 Python package, this is still practical when each of the roughly 40 input maps is ~10 Megabytes, but will not be if they are over 1 Gigabyte.

The solution to these issues was to reimplement PySM from scratch focusing of these features:

* Use the `numba` [@numba] Just-In-Time compiler for Python to reduce memory overhead and optimize performance: the whole integration loop of a template map over the frequency response of an instrument is performed in a single pass in automatically compiled and multi-threaded Python code.
* The target is to support template maps at a resolution of 0.4 arcminutes (HEALPix $N_{side}=8192$), this is difficult on a single node, so we use `mpi4py` to coordinate execution of PySM 3 using MPI across multiple machines.
* When running over MPI we cannot smooth the maps with the instrument beam via `healpy`, we need to rely on `libsharp` [@libsharp], a distributed implementation of spherical harmonic transforms.
* Input template maps are not included in the package, they are downloaded as needed and cached locally using the infrastructure provided by `astropy` [@astropy2013; @astropy2018].

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

At the moment it is not very useful to run at resolutions higher than $N_{side}=512$ because there is no actual template signal at smaller scales. However, it demonstrates the performance improvements that will make working with higher resolution templates possible.

# Future work

PySM 3 opens the way to implement a new category of models at much higher resolution. However, instead of just upgrading the current models to smaller scales we want to also update them with the latest knowledge of Galactic emission and gather feedback from each of the numerous CMB experiments. For this reason we are collaborating with the Panexperiment Galactic Science group to lead the development of the new class of models to be included in PySM 3.

# How to cite

If you are using PySM 3 for your work, please cite this paper for the software itself; for the actual emission modelling please also cite the original PySM 2 paper [@pysm17]. There will be a future paper on the generation of new PySM 3 astrophysical models.

# Acknowledgements

We acknowledge NASA for supporting Andrea Zonca's work on the project through the funding of the grant Theoretical & Computational Astrophysics Network `80NSSC18K1487`.

# References
