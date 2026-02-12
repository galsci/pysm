Cache Utilities
===============

PySM includes helper utilities to cache expensive preset map evaluations to compressed
``.npz`` files. This is useful in notebooks and exploratory workflows where models like
``d10`` and ``s5`` are repeatedly loaded at low ``nside``.

Functions
---------

- :py:func:`pysm3.get_or_create_cached_preset_map` computes a preset map in-process and caches it.
- :py:func:`pysm3.get_or_create_cached_preset_map_subprocess` computes a preset map in a subprocess,
  then loads it from cache. This is safer for notebook kernels when loading heavy templates.
  If the subprocess is terminated by the environment, it automatically falls back
  to in-process generation.

Example
-------

.. code-block:: python

    from pathlib import Path

    import pysm3
    import pysm3.units as u

    cache_dir = Path("docs/.cache")

    d10_map = pysm3.get_or_create_cached_preset_map_subprocess(
        preset_string="d10",
        nside=64,
        freq=145 * u.GHz,
        cache_dir=cache_dir,
    )

    s5_map = pysm3.get_or_create_cached_preset_map_subprocess(
        preset_string="s5",
        nside=64,
        freq=30 * u.GHz,
        cache_dir=cache_dir,
    )

Cache files are named ``{preset}_{nside}_{freq:.1f}GHz.npz`` and contain an IQU map array
under the key ``map``.

Notebook Workflow
-----------------

For the Gaussian foreground comparison notebook, preprocess cache files once
from the terminal:

.. code-block:: bash

    source .venv/bin/activate
    python docs/generate_gaussian_fg_cache.py --nside 64

Then run ``docs/gaussian_fg_comparison.ipynb``. The notebook only performs
analysis/plotting and does not generate heavy reference caches.

Script options:

- ``--presets d10 s5`` selects which preset caches to build (defaults to ``d10 s5``).
- ``--force`` overwrites existing cache files.

Notebook reference tags can be overridden via environment variables:

.. code-block:: bash

    PYSM_NOTEBOOK_DUST_REF=gd0 PYSM_NOTEBOOK_SYNC_REF=gs0
