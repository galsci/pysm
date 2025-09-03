import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freqs = [
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
]

cat = h5py.File("data/matched_catalogs_2/catalog_100.0.h5", "r")

cutoff_flux = 1e-3

high_flux_sources_mask = cat["flux"][:] > cutoff_flux

(high_flux_sources_mask).sum()

high_flux_sources_mask.mean() * 100

for k, v in cat.items():
    print(k, v[:3])

(all_indices,) = np.nonzero(high_flux_sources_mask)

len(all_indices)

all_indices = np.array(sorted(all_indices))

all_indices

import pandas as pd
import xarray as xr

columns = ["theta", "phi", "flux", "polarized flux"]

flux = xr.DataArray(
    data=np.zeros((len(all_indices), len(freqs)), dtype=np.float64),
    coords={"index": all_indices, "freq": list(map(float, freqs))},
    name="flux",
)
fluxnorm = flux.copy()

polarized_flux = flux.copy()

sources_xr = xr.Dataset(
    {"flux": flux, "polarized_flux": polarized_flux, "fluxnorm": fluxnorm}
)
for freq in freqs:
    print(freq)
    cat = h5py.File(f"data/matched_catalogs_2/catalog_{freq}.h5", "r")
    for column in ["flux", "polarized_flux"]:
        sources_xr[column].loc[dict(index=all_indices, freq=float(freq))] = cat[
            column.replace("_", " ")
        ][high_flux_sources_mask]

sources_xr

min_idx = int(sources_xr.flux.sel(freq=100).argmin().item())
max_idx = int(sources_xr.flux.sel(freq=100).argmax().item())
print("Index with minimum flux at 100 GHz:", min_idx)
print("Index with maximum flux at 100 GHz:", max_idx)

sources_xr = sources_xr.sortby(sources_xr.flux.sel(freq=100.0), ascending=False)

len(sources_xr)

max_loc = int(sources_xr.flux.sel(freq=100).argmax().item())
min_loc = int(sources_xr.flux.sel(freq=100).argmin().item())
print("Location (position) of maximum flux at 100 GHz:", max_loc)
print("Location (position) of minimum flux at 100 GHz:", min_loc)

sources_xr["fluxnorm"] = sources_xr["flux"] / sources_xr["flux"].sel(freq=100)

sources_xr["logpolycoefflux"] = xr.DataArray(
    np.zeros((len(all_indices), 5), dtype=np.float64),
    dims=["index", "power"],
    coords={"power": np.arange(5)[::-1]},
)
sources_xr["logpolycoefnorm"] = sources_xr["logpolycoefflux"].copy()
sources_xr["logpolycoefpolflux"] = sources_xr["logpolycoefflux"].copy()

from scipy.optimize import curve_fit
import time


def model(freq, a, b, c, d, e):
    log_freq = np.log(freq)
    return a * log_freq ** 4 + b * log_freq ** 3 + c * log_freq ** 2 + d * log_freq + e


start_time = time.time()
total = len(sources_xr.coords["index"])
indices = sources_xr.coords["index"]
iterator = indices

for i, s in enumerate(iterator):
    sources_xr["logpolycoefflux"].loc[dict(index=s)] = curve_fit(
        model, sources_xr.coords["freq"], sources_xr.flux.sel(index=s)
    )[0]
    sources_xr["logpolycoefpolflux"].loc[dict(index=s)] = curve_fit(
        model, sources_xr.coords["freq"], sources_xr.polarized_flux.sel(index=s)
    )[0]
    sources_xr["logpolycoefnorm"].loc[dict(index=s)] = curve_fit(
        model, sources_xr.coords["freq"], sources_xr.fluxnorm.sel(index=s)
    )[0]
    if (i + 1) % 100 == 0 or (i + 1) == total:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")
        msg = f"ETA: {int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
        print(f"Processed {i+1}/{total} | {msg}")

sources_xr.logpolycoefflux.min(), sources_xr.logpolycoefflux.max()

output_catalog = sources_xr[["logpolycoefflux", "logpolycoefpolflux"]]

output_catalog.logpolycoefflux.attrs["units"] = "Jy"
output_catalog.logpolycoefpolflux.attrs["units"] = "Jy"

for coord in ["theta", "phi"]:
    output_catalog = output_catalog.assign_coords(
        **{coord: (("index"), cat[coord][high_flux_sources_mask].astype(np.float64))}
    )

output_filename = "data/websky_high_flux_catalog_1mJy.h5"

output_catalog.coords["theta"].attrs["units"] = "rad"
output_catalog.coords["phi"].attrs["units"] = "rad"
output_catalog.coords["theta"].attrs["reference_frame"] = "Galactic"
output_catalog.coords["phi"].attrs["reference_frame"] = "Galactic"

output_catalog.attrs["description"] = (
    "Websky catalog of sources with flux > 1 mJy at 100 GHz, fitted with a 4th order polynomial in log frequency. "
    "Galactic reference frame. Sorted by flux at 100 GHz (descending). "
    "The 'index' coordinate gives the original index in the Websky catalog."
)

output_catalog.to_netcdf(
    output_filename, format="NETCDF4", mode="w"
)  # requires netcdf4 package

import xarray

xarray.open_dataset(output_filename)

import h5py

f = h5py.File(output_filename, "r")
f["logpolycoefflux"]

f["logpolycoefflux"].attrs["units"]
