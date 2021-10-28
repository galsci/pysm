|CI Tests| |Documentation Status| |PyPI| |Conda| |Astropy| |JOSS|

PySM 3
======

PySM generates full-sky simulations of Galactic emissions in intensity
and polarization relevant to CMB experiments. It is a large refactor of
`PySM 2 <https://github.com/bthorne93/PySM_public>`__ focused on
reducing memory usage, improving performance and run in parallel with
MPI.

See the documentation at https://pysm3.readthedocs.io

See changes in ``CHANGES.rst`` in the repository.

Related scientific papers
-------------------------

See `CITATION <https://github.com/galsci/pysm/blob/main/CITATION>`_

* `The Python Sky Model 3 software (Zonca et al, 2021) <https://arxiv.org/abs/2108.01444>`_
* `The Python Sky Model: software for simulating the Galactic microwave sky (Thorne et al, 2017) <https://arxiv.org/abs/1608.02841>`_

Install
-------

See the `documentation <https://pysm3.readthedocs.io/en/latest/#installation>`_

* Install with ``pip install .`` or with ``pip install .[test]`` to also install the requirements for running tests
* Optionally, if you have an MPI environment available and you would like to test the MPI capabilities of PySM, install ``mpi4py`` and ``libsharp``, check the documentation link above for more details.
* Check code style with ``tox -e codestyle``
* Test with ``pytest`` or ``tox -e test``
* Building docs requires ``pandoc``, not the python package, the actual ``pandoc`` command line tool, install it with conda or your package manager
* Build docs locally with ``tox -e build_docs``

Support
-------

For any question or issue with the software `open an issue <https://github.com/galsci/pysm/issues/>`_.

Release
-------

* Tag the new version with git
* ``pip install build --upgrade``
* ``python -m build --sdist --wheel .``
* ``twine upload dist/*``

.. |CI Tests| image:: https://github.com/galsci/pysm/actions/workflows/ci_tests.yml/badge.svg
   :target: https://github.com/galsci/pysm/actions/workflows/ci_tests.yml
.. |Documentation Status| image:: https://readthedocs.org/projects/pysm3/badge/?version=latest
   :target: https://pysm3.readthedocs.io/en/latest/?badge=latest
.. |PyPI| image:: https://img.shields.io/pypi/v/pysm3
   :target: https://pypi.org/project/pysm3/
.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pysm3
   :target: https://anaconda.org/conda-forge/pysm3
.. |Astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
.. |JOSS| image:: https://joss.theoj.org/papers/8f2d6c3bbf6cbeffbb403a1207fa8de7/status.svg
   :target: https://joss.theoj.org/papers/8f2d6c3bbf6cbeffbb403a1207fa8de7
