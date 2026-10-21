#!/usr/bin/env python
# coding: utf-8

import os
from pixell import enmap, reproject

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
import logging

pysm3.set_verbosity()

catalog_filename = sys.argv[1]

nside = int(sys.argv[2])

output_path = os.path.dirname(catalog_filename) + f"/{nside}/"

car_map_resolution = None


if nside == 8192:
    car_map_resolution = hp.nside2resol(nside, arcmin=True) * u.arcmin / 1.3

freqs = (
    list(
        map(
            float,
            [
                "1.0",
                "10.0",
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
                "1000.0",
            ],
        )
    )
    * u.GHz
)

assert len(freqs) == 16

freq = freqs[int(os.environ.get("SLURM_ARRAY_TASK_ID", 3))]

out_filename = output_path + f"{freq.value:05.1f}.fits"

if os.path.exists(out_filename.replace(".h5", "COMPLETED.txt")):
    sys.exit(0)


catalog_size = len(h5py.File(catalog_filename)["theta"])

step = 1
# slice_size = int(28.2 * 1e6 * step)

fwhm = {8192: 0.9 * u.arcmin, 4096: 2.6 * u.arcmin, 2048: 5.1 * u.arcmin}

m = None
# for slice_start in range(0, catalog_size, slice_size):

completed_file = out_filename.replace(".fits", "_COMPLETED")

if os.path.exists(completed_file):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{completed_file} exists, skipping.")
    sys.exit(0)
else:
    gc.collect()
    catalog = PointSourceCatalog(
        catalog_filename,
        # catalog_slice=np.index_exp[slice_start : slice_start + slice_size : step],
        nside=nside,
    )
    temp_m = catalog.get_emission(
        freq,
        fwhm=fwhm[nside],
        car_map_resolution=car_map_resolution,
        return_car=True,
        return_healpix=False,
    )
    if m is None:
        m = temp_m
    else:
        m += temp_m
    del temp_m

    enmap.write_map(out_filename.replace(".fits", ".h5"), m, fmt="hdf")

    healpix_map = reproject.map2healpix(
        m,
        nside,
        method="spline",
    )

    hp.write_map(
        out_filename,
        healpix_map,
        column_units="uK_RJ",
        coord="G",
        overwrite=True,
    )

    open(completed_file, "w").close()
