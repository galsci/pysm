PySM Documentation
==================

PySM 3 generates full-sky simulations of Galactic foregrounds in intensity and polarization relevant for CMB experiments. The components simulated are: thermal dust, synchrotron, AME, free-free, and CMB at a given HEALPix $N_{side}$, with an option to integrate over a bandpass and to smooth with a given beam.

There is scope for a few options for the model for each component, attempting to be consistent with current data.

Currently much of the available data is limited in resolution at degree-scale. We therefore make efforts to provide reasonable small-scale simulations to extend the data to higher multipoles. The details of the procedures developed can be found in the accompanying paper.

If you are using this code please cite the PySM 2 paper `Thorne at al <https://arxiv.org/abs/1608.02841>`_

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

Best practices for model execution
----------------------------------

PySM 3, in order to have the same behaviour of PySM 2, uses `hp.ud_grade` to change the resolution of the map if the HEALPix $N_{side}$ of the input templates is different from the requested output resolution, specified in the emission class (subclass of :py:class:`~pysm3.Model`) or in the :py:class:`~pysm3.Sky` class.

`hp.ud_grade` generally creates artifacts in the spectra and should be avoided unless the specific application can tolerate that.

Therefore we recommend to execute all PySM 2 derived models (e.g. `d0` to `d8`) at their native $N_{side}$ of 512, and then use the `output_nside` parameter of :py:func:`apply_smoothing_and_coord_transform` to transform to the target resolution, whether higher or lower, in Spherical Harmonics domain.

PySM 3 native models (e.g. `d9` to `d11` and `s4` to `s6`) instead have precomputed templates (generated in Spherical Harmonics domain) from $N_{side}$ 2048 to 8192. Therefore we recommend to execute them at 2048 if the target $N_{side}$ is 1024 or lower and at 2*$N_{side}$ if the target is 2048 or higher, except 8192 which is the highest possible resolution.

There are some exceptions, so always check in the documentation or the `max_nside` parameter of the models, for example the `d12` model, even if native to PySM 3, uses externally provided maps which are only available at $N_{side}$=2048.

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


Installation
============

The easiest way to install the last release is to use `conda`::

    conda install -c conda-forge pysm3

or `pip`::

    pip install pysm3

See the `changelog on Github <https://github.com/galsci/pysm/blob/main/CHANGES.rst>`_ for details about what is included in each release.

Install at NERSC
----------------

Optionally replace with a newer anaconda environment::

    module load python/3.7-anaconda-2019.10
    conda create -c conda-forge -n pysm3 pysm3 python=3.7 ipython
    conda activate pysm3
    module unload python

Development install
-------------------

The development version is available in the `main` branch of the `GitHub repository <https://github.com/galsci/pysm>`_,
you can clone and install it with::

    pip install .

Create a development installation with::

    pip install -e .

Install the requirements for testing with::

    pip install -e .[test]

Execute the unit tests with::

    pytest

Configure verbosity
-------------------

PySM uses the `logging` module to configure its verbosity,
by default it will only print warnings and errors, to configure logging
you can access the "pysm3" logger with::

    import logging
    log = logging.getLogger("pysm3")

configure the logging level::

    log.setLevel(logging.DEBUG)

redirect the logs to the console::

    handler = logging.StreamHandler()
    log.addHandler(handler)

or customize their format::

    log_format="%(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

For more details see the `Python documentation <https://docs.python.org/3/library/logging.html>`_.

Tutorials
=========

All the tutorials are Jupyter Notebooks and can be accessed `from the repository <https://github.com/galsci/pysm/tree/main/docs>`_:

.. toctree::
  :maxdepth: 2

  basic_use
  model_data
  bandpass_integration
  smoothing_coord_rotation
  customize_components
  mpi

Contributing
============

.. toctree::
  :maxdepth: 2

  contributing

Reference/API
=============

.. automodapi:: pysm3
