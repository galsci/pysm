#!/usr/bin/env python
# coding: utf-8

import os
from pixell import enmap

# for jupyter.nersc.gov otherwise the code only uses 2 cores
os.environ["OMP_NUM_THREADS"] = "128"

import numpy as np
import healpy as hp
import pysm3
from pysm3 import units as u
from pysm3.models import PointSourceCatalog
import matplotlib.pyplot as plt
import xarray as xr
import h5py
import gc
import sys

pysm3.set_verbosity()

catalog_filename = "/pscratch/sd/z/zonca/websky_full_catalog_trasp.h5"

nside = int(sys.argv[1])

car_map_resolution = None

if nside == 8192:
    car_map_resolution = hp.nside2resol(nside, arcmin=True) * u.arcmin / 1.3

freqs = (
    list(
        map(
            float,
            [
                "5.0",
                "18.7",
                "24.5",
                "44.0",
                "70.0",
                "100.0",
                "143.0",
                "217.0",
                "353.0",
                "545.0",
                "643.0",
                "729.0",
                "857.0",
                "906.0",
            ],
        )
    )
    * u.GHz
)

freq = freqs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

out_filename = catalog_filename.replace(
    ".h5", f"_nside_{nside}_map_{freq.value:04.1f}.h5"
)


if os.path.exists(out_filename.replace(".h5", "COMPLETED.txt")):
    sys.exit(0)


catalog_size = len(h5py.File(catalog_filename)["theta"])

slice_size = int(2.82 * 1e6)

fwhm = {8192: 0.9 * u.arcmin, 4096: 2.6 * u.arcmin, 2048: 5.1 * u.arcmin}

for slice_start in range(0, catalog_size, slice_size):
    gc.collect()
    catalog = PointSourceCatalog(
        catalog_filename,
        catalog_slice=np.index_exp[slice_start : slice_start + slice_size],
        nside=nside,
    )
    if slice_start == 0:
        m = catalog.get_emission(
            freq,
            fwhm=fwhm[nside],
            car_map_resolution=car_map_resolution,
            return_car=True,
        )
    else:
        m += catalog.get_emission(
            freq,
            fwhm=fwhm[nside],
            car_map_resolution=car_map_resolution,
            return_car=True,
        )

enmap.write_map(out_filename, m, fmt="hdf")

open(out_filename.replace(".h5", "_COMPLETED"), "a").close()
