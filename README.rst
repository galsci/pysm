|Build Status| |Documentation Status| |PyPI| |Conda| |Astropy|

PySM 3
======

PySM generates full-sky simulations of Galactic emissions in intensity
and polarization relevant to CMB experiments. It is a large refactor of
`PySM 2 <https://github.com/bthorne93/PySM_public>`__ focused on
reducing memory usage, improving performance and run in parallel with
MPI.

See the documentation at https://pysm3.readthedocs.io

* Check code style with ``tox -e codestyle``
* Test with ``pytest`` or ``tox -e test``
* Build docs locally with ``tox -e build_docs``

See changes in ``CHANGES.rst`` in the repository.

Install
-------

See the `documentation <https://pysm3.readthedocs.io/en/latest/#installation>`_

Release
-------

Follow the `Astropy guide to release a new version <https://docs.astropy.org/en/stable/development/astropy-package-template.html>`.

.. |Build Status| image:: https://travis-ci.org/healpy/pysm.svg?branch=master
   :target: https://travis-ci.org/healpy/pysm
.. |Documentation Status| image:: https://readthedocs.org/projects/pysm3/badge/?version=latest
   :target: https://pysm3.readthedocs.io/en/latest/?badge=latest
.. |PyPI| image:: https://img.shields.io/pypi/v/pysm3
   :target: https://pypi.org/project/pysm3/
.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pysm3
   :target: https://anaconda.org/conda-forge/pysm3
.. |Astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
