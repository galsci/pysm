#!/usr/bin/env python
"""Create a high–flux Websky point–source mini catalog with spectral fits.

This script selects sources above a flux density threshold (default 1 mJy at a
given reference frequency, default 100 GHz) from a set of Websky-matched
catalog HDF5 files (one per frequency). For each selected source it fits a
degree-N polynomial (default degree 4) in log(frequency) to the total and
polarized flux densities across the provided frequencies. The resulting
polynomial coefficients and sky coordinates are stored in a compact NetCDF4
file suitable for use inside PySM or other simulators.

Key improvements over the original notebook-exported script:
 - Parameterized CLI (frequencies, cutoff, reference frequency, degree, output)
 - Faster linear least-squares fit (no iterative curve_fit loop)
 - Deterministic ordering (sorted by flux at reference frequency, descending)
 - Clean separation into functions & safer file handling (context managers)
 - Minimal dependencies (removed unused matplotlib/pandas/etc.)
 - Metadata & units preserved
 - Deterministic ordering (sorted by fitted model flux at reference frequency, descending)

Example:
  python websky_sources_high_flux_catalog_out_1mJy.py \
      --input-pattern 'data/matched_catalogs_2/catalog_{freq:.1f}.h5' \
      --frequencies 18.7 24.5 44.0 70.0 100.0 143.0 217.0 353.0 545.0 643.0 729.0 857.0 906.0 \
      --cutoff-mjy 1.0 --ref-freq 100.0 \
      --output data/websky_high_flux_catalog_1mJy.h5

Requires packages: numpy, xarray, h5py, netCDF4.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
import subprocess
import sys
import shlex
from pathlib import Path
from typing import List

import h5py
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reduced high-flux Websky catalog with spectral polynomial coefficients"
    )
    parser.add_argument(
        "--input-pattern",
        required=False,
        default="data/matched_catalogs_2/catalog_{freq:.1f}.h5",
        help="Pattern for per-frequency input HDF5 files. Must contain {freq}.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        type=float,
        default=[
            18.7,
            24.5,
            44.0,
            70.0,
            100.0,
            143.0,
            217.0,
            353.0,
            545.0,
            643.0,
            729.0,
            857.0,
            906.0,
        ],
        help="Frequencies in GHz (must match available files).",
    )
    parser.add_argument(
        "--cutoff-mjy",
        type=float,
        default=1.0,
        help="Flux density cutoff in mJy at the reference frequency.",
    )
    parser.add_argument(
        "--ref-freq",
        type=float,
        default=100.0,
        help="Reference frequency in GHz at which cutoff and sorting are applied (via fitted model).",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        help="Polynomial degree in log(frequency) for the spectral fit.",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="data/websky_high_flux_catalog_1mJy.h5",
        help="Output NetCDF filename (will be overwritten).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Progress print interval (number of sources).",
    )
    return parser.parse_args()


def open_catalog(filename: str | Path):
    if not Path(filename).exists():
        raise FileNotFoundError(f"Input catalog not found: {filename}")
    return h5py.File(filename, "r")


def select_high_flux_indices(
    ref_filename: str | Path, cutoff_jy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask and sorted indices of sources above cutoff.

    Parameters
    ----------
    ref_filename : str | Path
        HDF5 catalog filename at the reference frequency.
    cutoff_jy : float
        Flux density cutoff in Jy.
    """
    with open_catalog(ref_filename) as f:  # type: ignore[call-arg]
        # h5py stubs are loose; add ignores where needed
        flux = np.asarray(f["flux"][...], dtype=np.float64)  # type: ignore[index]
        mask = flux > cutoff_jy
        (indices,) = np.nonzero(mask)
    return mask, np.sort(indices)


def build_flux_dataset(
    indices: np.ndarray,
    mask: np.ndarray,
    freqs: List[float],
    input_pattern: str,
) -> xr.Dataset:
    """Load flux & polarized flux for selected sources across frequencies."""

    nsrc = len(indices)
    freq_arr = np.array(freqs, dtype=float)
    flux = xr.DataArray(
        np.zeros((nsrc, len(freqs)), dtype=np.float64),
        dims=("index", "freq"),
        coords={"index": indices, "freq": freq_arr},
        name="flux",
    )
    pol_flux = flux.copy(data=np.zeros_like(flux.data))

    have_polarized = True
    theta = phi = None  # will set on first iteration
    for fghz in freqs:
        filename = input_pattern.format(freq=fghz)
        with open_catalog(filename) as cat:  # type: ignore[call-arg]
            flux_vals = np.asarray(cat["flux"][...], dtype=np.float64)[
                mask
            ]  # type: ignore[index]
            flux.loc[dict(freq=fghz)] = flux_vals
            if "polarized flux" in cat:
                pol_flux_vals = np.asarray(
                    cat["polarized flux"][...], dtype=np.float64
                )[  # type: ignore[index]
                    mask
                ]
                pol_flux.loc[dict(freq=fghz)] = pol_flux_vals
            else:
                have_polarized = False
            if theta is None:
                theta = np.asarray(cat["theta"][...], dtype=np.float64)[mask]  # type: ignore[index]
                phi = np.asarray(cat["phi"][...], dtype=np.float64)[mask]  # type: ignore[index]

    ds = xr.Dataset({"flux": flux})
    if have_polarized:
        ds["polarized_flux"] = pol_flux
    else:  # drop placeholder if not present
        pol_flux = None  # type: ignore
    ds = ds.assign_coords(theta=("index", theta), phi=("index", phi))
    ds.theta.attrs.update(units="rad", reference_frame="Galactic")
    ds.phi.attrs.update(units="rad", reference_frame="Galactic")
    return ds


def fit_log_poly(ds: xr.Dataset, degree: int, progress_every: int = 200) -> xr.Dataset:
    """Vectorized polynomial fit in log(frequency) for (polarized) flux.

    Uses a single lstsq per (flux / polarized_flux) array instead of per-source loop.
    """
    freqs = ds.freq.values
    logf = np.log(freqs)
    deg = degree
    powers = np.arange(deg, -1, -1)
    A = np.vstack([logf**p for p in powers]).T  # (Nfreq, deg+1)

    # Arrange data as (Nfreq, Nsrc)
    Y_flux = ds.flux.transpose("freq", "index").values  # (Nfreq, Nsrc)
    t0 = time.time()
    X_flux = np.linalg.lstsq(A, Y_flux, rcond=None)[0]  # (deg+1, Nsrc)
    print(
        f"Flux fit: solved for {Y_flux.shape[1]} sources in {time.time()-t0:.2f}s (vectorized)"
    )

    # Store coefficients with dims (power, index) so np.polyval(coeffs, logf) works directly.
    ds["logpolycoefflux"] = xr.DataArray(
        X_flux,
        dims=("power", "index"),
        coords={"power": powers, "index": ds.index.values},
        attrs={"units": "Jy", "coeff_orientation": "power_first_highest_degree_first"},
    )

    if "polarized_flux" in ds:
        t1 = time.time()
        Y_pol = ds.polarized_flux.transpose("freq", "index").values
        X_pol = np.linalg.lstsq(A, Y_pol, rcond=None)[0]
        print(
            f"Polarized flux fit: solved for {Y_pol.shape[1]} sources in {time.time()-t1:.2f}s (vectorized)"
        )
        ds["logpolycoefpolflux"] = xr.DataArray(
            X_pol,
            dims=("power", "index"),
            coords={"power": powers, "index": ds.index.values},
            attrs={
                "units": "Jy",
                "coeff_orientation": "power_first_highest_degree_first",
            },
        )
    return ds


def evaluate_flux_at_logfreq(coeff_da: xr.DataArray, log_freq: float) -> xr.DataArray:
    """Evaluate polynomial at log_freq using numpy.polyval.

    Expects coeff_da with dims (power, index) where 'power' is ordered from
    highest degree to 0. This matches how we store coefficients, enabling:
        np.polyval(coeff_da.values, log_freq)
    to yield an array of shape (Nsrc,).
    """
    coeff_aligned = coeff_da.transpose("power", "index")  # ensure ordering
    coeffs = coeff_aligned.values  # (deg+1, Nsrc)
    fitted = np.polyval(coeffs, log_freq)  # (Nsrc,)
    return xr.DataArray(fitted, dims=("index",), coords={"index": coeff_da.index})


def main():
    args = parse_args()
    freqs = args.frequencies
    ref_freq = args.ref_freq
    if ref_freq not in freqs:
        raise ValueError("Reference frequency must be included in --frequencies list")
    cutoff_jy = args.cutoff_mjy * 1e-3  # mJy -> Jy

    ref_filename = args.input_pattern.format(freq=ref_freq)
    print(f"Selecting sources from {ref_filename} with cutoff {cutoff_jy} Jy")
    mask, indices = select_high_flux_indices(ref_filename, cutoff_jy)
    print(f"Selected {len(indices)} sources above threshold.")

    ds = build_flux_dataset(indices, mask, freqs, args.input_pattern)

    # Fit spectral polynomial coefficients first
    ds = fit_log_poly(ds, args.degree, progress_every=args.progress_every)

    # Sort by fitted (modeled) flux at the reference frequency (descending) using np.polyval (no storage)
    log_ref = np.log(ref_freq)
    sort_key = evaluate_flux_at_logfreq(ds.logpolycoefflux, log_ref)
    ds = ds.sortby(sort_key, ascending=False)

    base_vars = ["logpolycoefflux"]
    if "logpolycoefpolflux" in ds:
        base_vars.append("logpolycoefpolflux")
    ds_out = ds[base_vars]

    ds_out.attrs["description"] = (
        f"Websky sources with flux > {args.cutoff_mjy} mJy at {ref_freq} GHz. "
        f"Polynomial degree {args.degree} fit in log(frequency) for flux (and polarized flux if available). "
        "Frequencies in GHz. Ordered by descending fitted flux at reference frequency (polynomial evaluated at ref freq). "
        "'index' gives original catalog index before selection/sorting. "
        "NOTE: Coefficient arrays are stored transposed relative to PySM 3.4.1 and 3.4.2 (which used (index,power)). "
        "Now dims are (power,index) with highest degree first, enabling direct "
        "numpy.polyval(coeffs, log_freq) without any transpose step."
    )
    ds_out.attrs["reference_frequency_GHz"] = ref_freq
    ds_out.attrs["flux_cutoff_mJy"] = args.cutoff_mjy
    ds_out.attrs["polynomial_degree"] = args.degree
    ds_out.attrs["sorted_by"] = (
        "polyval(logpolycoefflux, log(ref_freq)) with coeff dims (power,index)"
    )
    # Reference frame for theta/phi coordinates
    ds_out.attrs["ref_frame"] = "Galactic"
    # Generation timestamp in UTC ISO8601
    ds_out.attrs["generated_utc"] = datetime.now(timezone.utc).isoformat()
    # Git commit hash (required)
    script_dir = Path(__file__).resolve().parent
    repo_root = None
    for p in [script_dir] + list(script_dir.parents):
        if (p / ".git").exists():
            repo_root = p
            break
    if repo_root is None:
        raise RuntimeError(
            "Cannot determine git repository root (.git not found); aborting."
        )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to obtain git commit hash: {e}") from e
    if not commit:
        raise RuntimeError("Empty git commit hash returned; aborting.")
    ds_out.attrs["git_commit"] = commit
    # Full command invocation
    ds_out.attrs["command"] = " ".join(shlex.quote(a) for a in sys.argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ...")
    ds_out.to_netcdf(output_path, format="NETCDF4", mode="w")
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    # Allow import without executing; run only when invoked as script
    main()
