|Build Status| |Documentation Status|

PySM 3
======

PySM generates full-sky simulations of Galactic emissions in intensity
and polarization relevant to CMB experiments. It is a large refactor of
`PySM 2 <https://github.com/bthorne93/PySM_public>`__ focused on
reducing memory usage, improving performance and run in parallel with
MPI.

See the documentation at https://pysm3.readthedocs.io

Install
-------

Requirements
~~~~~~~~~~~~

See ```requirements.txt`` <requirements.txt>`__

Conda
~~~~~

::

   conda install -c conda-forge pysm3

See the `conda repository <https://anaconda.org/conda-forge/pysm3>`__
and the `feedstock <https://github.com/conda-forge/pysm3-feedstock>`__

Pip
~~~

::

   pip install pysm3

Libsharp
~~~~~~~~

MPI-Distributed smoothing (optional) requires ``libsharp``, it is
easiest to install the conda package:

::

   conda install -c conda-forge libsharp=*=*openmpi*

It also has a ``mpich`` version:

::

   conda install -c conda-forge libsharp=*=*mpich*

.. |Build Status| image:: https://travis-ci.org/healpy/pysm.svg?branch=master
   :target: https://travis-ci.org/healpy/pysm
.. |Documentation Status| image:: https://readthedocs.org/projects/pysm3/badge/?version=latest
   :target: https://pysm3.readthedocs.io/en/latest/?badge=latest
