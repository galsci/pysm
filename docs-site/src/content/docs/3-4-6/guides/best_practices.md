---
title: Best practices
description: How to run PySM 3 models at the correct native resolution to avoid artifacts.
---

## Run models at their native Nside

PySM 3, in order to have the same behaviour of PySM 2, uses `hp.ud_grade` to change the resolution of the map if the HEALPix $N_{side}$ of the input templates is different from the requested output resolution, specified in the emission class (`pysm3.Model`) or in the `pysm3.Sky` class.

`hp.ud_grade` generally creates artifacts in the spectra and should be avoided unless the specific application can tolerate that.

Therefore we recommend to execute all PySM 2 derived models (e.g. `d0` to `d8`) at their native $N_{side}$ of 512, and then use the `output_nside` parameter of `pysm3.apply_smoothing_and_coord_transform` to transform to the target resolution, whether higher or lower, in Spherical Harmonics domain.

PySM 3 native models (e.g. `d9` to `d11` and `s4` to `s6`) instead have precomputed templates (generated in Spherical Harmonics domain) from $N_{side}$ 2048 to 8192. Therefore we recommend to execute them at 2048 if the target $N_{side}$ is 1024 or lower and at $2 \times N_{side}$ if the target is 2048 or higher, except 8192 which is the highest possible resolution.

There are some exceptions, so always check in the documentation or the `max_nside` parameter of the models, for example the `d12` model, even if native to PySM 3, uses externally provided maps which are only available at $N_{side}$=2048.

## Dependencies

PySM is written in pure Python, multi-threading and Just-In-Time compilation are provided by `numba`. The required packages are:

- `healpy`
- `numba`
- `toml`
- `astropy`

How many threads are used by `numba` can be controlled by setting:

```bash
export OMP_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
```

For debugging purposes, you can also completely disable the `numba` JIT compilation and have the code just use plain Python, which is slower, but easier to debug:

```bash
export NUMBA_DISABLE_JIT=1
```

## Configuring verbosity

PySM uses the `logging` module to configure its verbosity, by default it will only print warnings and errors.

A log of useful information, for example intermediate results and timing can be accessed enabling the `INFO` level of logging. We provide a simplified function to configure this:

```python
pysm3.set_verbosity()
```

By default this sets verbosity to `INFO`, otherwise you can specify a level:

```python
pysm3.set_verbosity(logging.DEBUG)
```

To configure logging you can access the `pysm3` logger with:

```python
import logging
log = logging.getLogger("pysm3")
```
