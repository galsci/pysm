PySM Documentation
==================

This code generates full-sky simulations of Galactic foregrounds in intensity and polarization relevant for CMB experiments. The components simulated are: thermal dust, synchrotron, AME, free-free, and CMB at a given Nside, with an option to integrate over a top hat bandpass, to add white instrument noise, and to smooth with a given beam.

There is scope for a few options for the model for each component, attempting to be consistent with current data.

Currently much of the available data is limited in resolution at degree-scale. We therefore make efforts to provide reasonable small-scale simulations to extend the data to higher multipoles. The details of the procedures developed can be found in the accompanying paper.

This code is based on the large-scale Galactic part of `Planck Sky Model <http://www.apc.univ-paris7.fr/~delabrou/PSM/psm.html>`_ (`Delabrouille 2012 <https://arxiv.org/abs/1207.3675>`_) code and uses some of its inputs.

Models
------

Each model is identified with a letter and a number, the letter indicates the kind of emission and the number the type of model, generally in order of complexity starting at 1. For example for dust we start with `d1` based on Planck commander results, `d4` has 2 dust populations and `d6` implements a model of dust frequency decorrelation.

For example free-free:

    **f1**: We model the free-free emission using the analytic model assumed in the Commander fit to the Planck 2015 data (Draine 2011 Physics of the Interstellar and Intergalactic Medium) to produce a degree-scale map of free-free emission at 30 GHz. We add small scales to this using a procedure outlined in the accompanying paper. This map is then scaled in frequency by applying a spatially constant power law index of -2.14.

* See the PySM 2 paper `Thorne at al <https://arxiv.org/abs/1608.02841>`_
* See the documentation about :ref:`models`

.. toctree::
  :maxdepth: 2

  models

Dependencies
============

PySM is written in Python and uses the healpy, numpy, scipy, numba and astropy packages. Optionally, it supports mpi4py.

Installation
============

Clone the `GitHub repository <https://github.com/healpy/pysm>`_ and run::

    pip install .

for a development installation, instead run::

    pip install -e .

Tutorials
=========

.. toctree::
  :maxdepth: 2

  basic_use
  model_data
  bandpass_integration
  smoothing_coord_rotation
  customize_components

Reference/API
=============

.. automodapi:: pysm
