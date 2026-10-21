---
title: "Best practices for model execution"
description: "PySM 3, in order to have the same behaviour of PySM 2, uses `hp.ud_grade` to change the resolution of the map if the HEALPix $N_{side}$ of the input templates i"
---
PySM 3, in order to have the same behaviour of PySM 2, uses `hp.ud_grade` to change the resolution of the map if the HEALPix $N_{side}$ of the input templates is different from the requested output resolution, specified in the emission class (subclass of `pysm3.Model`) or in the `pysm3.Sky` class.

`hp.ud_grade` generally creates artifacts in the spectra and should be avoided unless the specific application can tolerate that.

Therefore we recommend to execute all PySM 2 derived models (e.g. `d0` to `d8`) at their native $N_{side}$ of 512, and then use the `output_nside` parameter of `apply_smoothing_and_coord_transform` to transform to the target resolution, whether higher or lower, in Spherical Harmonics domain.

PySM 3 native models (e.g. `d9` to `d11` and `s4` to `s6`) instead have precomputed templates (generated in Spherical Harmonics domain) from $N_{side}$ 2048 to 8192. Therefore we recommend to execute them at 2048 if the target $N_{side}$ is 1024 or lower and at 2\*$N_{side}$ if the target is 2048 or higher, except 8192 which is the highest possible resolution.

There are some exceptions, so always check in the documentation or the `max_nside` parameter of the models, for example the `d12` model, even if native to PySM 3, uses externally provided maps which are only available at $N_{side}$=2048.
