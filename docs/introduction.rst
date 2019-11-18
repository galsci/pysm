Introduction
************

This code generates full-sky simulations of Galactic foregrounds in intensity and polarization relevant for CMB experiments. The components simulated are: thermal dust, synchrotron, AME, free-free, and CMB at a given Nside, with an option to integrate over a top hat bandpass, to add white instrument noise, and to smooth with a given beam.

There is scope for a few options for the model for each component, attempting to be consistent with current data. The current v-1.0 version has typically two-three options for each component.

Currently much of the available data is limited in resolution at degree-scale. We therefore make efforts to provide reasonable small-scale simulations to extend the data to higher multipoles. The details of the procedures developed can be found in the accompanying paper.

This code is based on the large-scale Galactic part of `Planck Sky Model <http://www.apc.univ-paris7.fr/~delabrou/PSM/psm.html>`_ (`Delabrouille 2012 <https://arxiv.org/abs/1207.3675>`_) code and uses some of its inputs.

Dependencies
============

PySM is written in Python and uses the healpy, numpy, scipy, and astropy packages. It is known to work with:

- python 2.7.6
- healpy 1.10.3
- numpy 1.12.1
- scipy 0.19.0

Installation
============

Clone the `GitHub repository <https://github.com/bthorne93/PySM_public>`_ and run::
  
  [sudo] python setup.py install [--user]
  
After this you can run the provided unit tests from the same directory::
  
  nosetests
  
Then you may import PySM in the standard way in a Python environment::
  
  import pysm

 
