|CI Tests| |Documentation Status| |PyPI| |Conda| |Astropy| |JOSS|

PySM 3
======

PySM generates full-sky simulations of Galactic emissions in intensity
and polarization relevant to CMB experiments. It is a large refactor of
`PySM 2 <https://github.com/bthorne93/PySM_public>`__ focused on
reducing memory usage, improving performance and run in parallel with
MPI.

See the documentation at https://pysm3.readthedocs.io

Contributor guidelines, coding standards, and testing expectations are documented in ``AGENTS.md`` (Repository Guidelines).

See changes in ``CHANGES.rst`` in the repository.

Related scientific papers
-------------------------

See `CITATION <https://github.com/galsci/pysm/blob/main/CITATION>`_

* `Full-sky Models of Galactic Microwave Emission and Polarization at Subarcminute Scales for the Python Sky Model (The PanEx GS Group, ApJ 991, 23, 2025) <https://iopscience.iop.org/article/10.3847/1538-4357/adf212>`_
* `The Python Sky Model 3 software (Zonca et al, 2021) <https://arxiv.org/abs/2108.01444>`_
* `The Python Sky Model: software for simulating the Galactic microwave sky (Thorne et al, 2017) <https://arxiv.org/abs/1608.02841>`_

Install
-------

See the `documentation <https://pysm3.readthedocs.io/en/latest/#installation>`_

* Install with ``pip install .`` or with ``pip install .[test]`` to also install the requirements for running tests
* Optionally, if you have an MPI environment available and you would like to test the MPI capabilities of PySM, install ``mpi4py`` and ``libsharp``, check the documentation link above for more details.
* This repository uses Git LFS for Jupyter notebook files (*.ipynb). Ensure Git LFS is installed and run ``git lfs pull`` after cloning to fetch large files.
* Check code style with ``uv run flake8 src/pysm3 --count --max-line-length=100``
* Test with ``uv run pytest -v``
* Building docs requires ``pandoc``, not the python package, the actual ``pandoc`` command line tool, install it with conda or your package manager
* Build docs locally with ``uv run sphinx-build -W -b html docs docs/_build/html``

Support
-------

For any question or issue with the software `open an issue <https://github.com/galsci/pysm/issues/>`_.

Release
-------

1. Review ``CHANGES.rst`` and move the entries you want to ship out of the
   ``Unreleased`` section into a dated ``<version> (<YYYY-MM-DD>)`` heading.
   Commit the changelog update (and any other release-related changes).
2. Ensure the working tree is clean and up to date with ``git status`` and
   ``git pull``.
3. Create or refresh a local environment using ``uv``::

       uv venv .venv
       uv pip install --python .venv/bin/python pip hatch

   Activate it for the remaining steps with ``source .venv/bin/activate``.
4. Run the test suite (at least ``pytest``) to verify the release build.
5. Create the annotated release tag, e.g. ``git tag -a 3.4.3 -m "Release 3.4.3"``.
6. Confirm Hatch picks up the tagged version::

       hatch version

   The output should match the tag (no ``.dev`` suffix).
7. Build the distribution artifacts::

       hatch build

8. Publish to PyPI using your API token. Hatch reads credentials from
   ``~/.pypirc`` (username ``__token__``). Alternatively export
   ``HATCH_INDEX_USER=__token__`` and ``HATCH_INDEX_AUTH=<pypi-token>`` before
   running::

       hatch publish --no-prompt

9. Push the tag (and any commits) to GitHub::

       git push --tags

10. Draft the GitHub release notes referencing the matching ``CHANGES.rst``
    entry and announce the release as needed.

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
