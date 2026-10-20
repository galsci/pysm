Installation
============

The easiest way to install the last release is to use `conda`::

    conda install -c conda-forge pysm3

or `pip`::

    pip install pysm3

Model data files are downloaded from remote servers on first use.

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
by default it will only print warnings and errors.

A log of useful information, for example intermediate results and timing can be
accessed enabling the `INFO` level of logging.
We provide a simplified function to configure this.

    pysm3.set_verbosity()

By default this sets verbosity to `INFO`, otherwise you can specify a level:

    pysm3.set_verbosity(logging.DEBUG)

To configure logging
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
