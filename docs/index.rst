PySM Documentation
==================

PySM 3 generates full-sky simulations of Galactic foregrounds in intensity and polarization relevant for CMB experiments. The components simulated are: thermal dust, synchrotron, AME, free-free, and CMB at a given HEALPix $N_{side}$, with an option to integrate over a bandpass and to smooth with a given beam.

There is scope for a few options for the model for each component, attempting to be consistent with current data.

Currently much of the available data is limited in resolution at degree-scale. We therefore make efforts to provide reasonable small-scale simulations to extend the data to higher multipoles. The details of the procedures developed can be found in the accompanying paper.

This code is based on the large-scale Galactic part of `Planck Sky Model <http://www.apc.univ-paris7.fr/~delabrou/PSM/psm.html>`_ (`Delabrouille 2012 <https://arxiv.org/abs/1207.3675>`_) code and uses some of its inputs.

Acknowledgement
---------------

If you are using this code please cite:

* `The Python Sky Model: software for simulating the Galactic microwave sky (Thorne et al, 2017) <https://arxiv.org/abs/1608.02841>`_
* `The Python Sky Model 3 software (Zonca et al, 2021) <https://arxiv.org/abs/2108.01444>`_
* `Full-sky Models of Galactic Microwave Emission and Polarization at Subarcminute Scales for the Python Sky Model (The PanEx GS Group, ApJ 991, 23, 2025) <https://iopscience.iop.org/article/10.3847/1538-4357/adf212>`_  :doc:`Author contributions statement <pysm_methods_author_contributions>`.

See the Bibtex format file `CITATION <https://github.com/galsci/pysm/blob/main/CITATION>`_

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
  pre_applied_beam

Dependencies
============

PySM is written in pure Python, multi-threading and Just-In-Time compilation are provided by `numba`.
The required packages are:

* `healpy`
* `numba`
* `toml`
* `astropy`
* `importlib_metadata` just for Python 3.7

How many threads are used by `numba` can be controlled by setting::

    export OMP_NUM_THREADS=2
    export NUMBA_NUM_THREADS=2

For debugging purposes, you can also completely disable the `numba` JIT compilation and have the code just use plain
Python, which is slower, but easier to debug::

    export NUMBA_DISABLE_JIT=1

Run PySM with MPI support
-------------------------

PySM 3 is capable of running across multiple nodes using a MPI communicator.

See the details in the `MPI section of the tutorial <mpi.ipynb>`_.

In order to run in parallel with MPI, it also needs a functioning MPI environment and:

* `mpi4py`

MPI-Distributed smoothing (optional) requires `libsharp`, it is easiest to install the conda package::

    conda install -c conda-forge libsharp=*=*openmpi*

It also has a `mpich` version::

    conda install -c conda-forge libsharp=*=*mpich*


Tutorials
=========

All the tutorials are Jupyter Notebooks and can be accessed `from the repository <https://github.com/galsci/pysm/tree/main/docs>`_:

.. toctree::
  :maxdepth: 2

  basic_use
  model_data
  bandpass_integration
  bandpass_sampling
  bandpass_sampling_comparison
  smoothing_coord_rotation
  customize_components
  mpi

Template preprocessing notebooks
================================

The following notebooks document the data-processing workflow used to build the
high-resolution templates shipped with PySM.

.. toctree::
  :maxdepth: 1

  Commander dust preprocessing <preprocess-templates/reproduce_pysm2_dust_pol>
  WMAP synchrotron preprocessing <preprocess-templates/reproduce_pysm2_sync_pol>
  Small-scale dust generation overview <preprocess-templates/small_scale_dust_pysm3>
  Generate dust small-scale artifacts <preprocess-templates/small_scale_dust_pysm3_generate_artifacts>
  Generate PySM3 dust templates <preprocess-templates/small_scale_dust_pysm3_generate_templates>
  Synchrotron beta small-scale modelling <preprocess-templates/synchrotron_beta>
  Synchrotron curvature modelling <preprocess-templates/synchrotron_curvature>
  Log-pol-tens formalism notebook <preprocess-templates/synchrotron_template_logpoltens>
  WebSky bright source catalog workflow <preprocess-templates/websky_sources_high_flux_catalog>
  Compare WebSky and Agora SZ templates <preprocess-templates/verify_templates/compare_websky_agora_sz>
  Compare WebSky, Agora, and FLAMINGO kSZ templates <preprocess-templates/verify_templates/compare_flamingo_ksz>
  Verify dust templates <preprocess-templates/verify_templates/verify_templates_dust>
  Verify synchrotron templates <preprocess-templates/verify_templates/verify_templates_synch>

Catalog generation notebooks
============================

.. toctree::
  :maxdepth: 1

  Create WebSky background catalog with Dask <preprocess-templates/catalog/background_create_websky_catalog_dask>
  Assemble WebSky background catalog with Dask <preprocess-templates/catalog/background_dask_assemble_catalog>
  Compare catalog and original WebSky sources <preprocess-templates/catalog/compare_catalog_to_original_websky>
  Inspect WebSky high-flux catalog output <preprocess-templates/catalog/websky_sources_high_flux_catalog_out_1mJy>

Template utility notebooks
==========================

.. toctree::
  :maxdepth: 1

  GNILC dust spectral index and temperature maps <preprocess-templates/gnilc_dust_spectralindex_Tdust>
  Generate GNILC spectral index and temperature maps <preprocess-templates/utils_gnilc_generate_map_spectralindex_Td>
  Generate synchrotron amplitude map <preprocess-templates/utils_synch_generate_map>
  Compute synchrotron curvature harmonic coefficients <preprocess-templates/utils_synch_generate_map_curvature>
  Generate synchrotron spectral index map <preprocess-templates/utils_synch_generate_map_spectralindex>

Contributing
============

.. toctree::
  :maxdepth: 2

  contributing

Supplementary material
======================

.. toctree::
  :maxdepth: 1

  websky
  pysm_methods_author_contributions

Reference/API
=============

.. automodapi:: pysm3
