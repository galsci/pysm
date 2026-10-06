#!/usr/bin/env python
"""Convert an Agora extragalactic radio source catalog to the PySM
``PointSourceCatalog`` HDF5 format.

Reads one of the Agora radio catalogs (Omori 2024, arXiv:2212.07420),
distributed through the Agora Globus collection under ``components/rad/``:

* lensed: ``agora_radiocat_len_universemachine_trinity_95_150_220ghz_
  randflux_datta2018_truncgauss.0.fits`` (source positions in ``RAL``/``DECL``)
* unlensed: ``agora_radiocat_unl_universemachine_trinity_95_150_220ghz_
  randflux_datta2018_truncgauss.0.fits`` (source positions in ``RA``/``DEC``)
Both catalogs contain 56,108,888 sources with redshift, black hole mass,
total flux density ``I5/I95/I150/I220`` in Jy, Stokes parameters
``Q/U`` at 95/150/220 GHz and the per-source polarization angle ``psi``.

The output follows the format read by
:py:class:`pysm3.models.catalog.PointSourceCatalog` (the model behind the
``rg2`` WebSky preset): datasets ``theta``/``phi`` in radians and
``logpolycoefflux``/``logpolycoefpolflux`` in Jy with shape
``(n_coeff, n_sources)``, highest degree first. The flux model is a
polynomial in the natural logarithm of frequency (GHz) whose value is the
flux density in Jy directly, clipped at 0 when negative:

.. math:: S(\\nu) = \\max\\left(0, \\sum_i c_i (\\ln \\nu)^{n-1-i}\\right)

Coefficients are computed exactly through the measured flux nodes by
inverting the shared Vandermonde matrix of the log-frequency nodes, so the
model reproduces the catalog fluxes to machine precision at the nodes:

* intensity: 4 coefficients through I at 5, 95, 150, 220 GHz (the 5 GHz
  anchor improves the low-frequency extrapolation)
* polarization: 3 coefficients through P = sqrt(Q^2 + U^2) at 95, 150,
  220 GHz

The per-source polarization angle ``psi`` (radians) is stored in the output
as well: the current ``PointSourceCatalog`` implementation draws random
polarization angles, so the stored angles allow a future extension to
reproduce the exact Agora Q/U. The conversion was validated against the
official Agora SPT-3G band maps, see
``docs/preprocess-templates/verify_templates/compare_websky_agora_radio.ipynb``.

Known issue (September 2025 data release): in the lensed catalog
``Q220 == U220`` for all sources, so the polarized flux of the lensed
catalog at 220 GHz is corrupted (the official 220 GHz map shares the bug).
The unlensed catalog is clean. See galsci/pysm#250.

Example:
    python agora_radio_sources_catalog.py \\
        --input agora_radiocat_unl_universemachine_trinity_95_150_220ghz_...fits \\
        --output agora_radio_unl_pysm.h5

Requires: numpy, astropy, h5py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

INTENSITY_FREQS_GHZ = [5.0, 95.0, 150.0, 220.0]
POLARIZED_FREQS_GHZ = [95.0, 150.0, 220.0]
REFERENCE_FREQ_GHZ = 150.0


def poly_coeffs_through_points(freqs_ghz, flux):
    """Exact polynomial coefficients through the given flux nodes.

    Parameters
    ----------
    freqs_ghz : np.array
        Frequencies of the flux measurements in GHz
    flux : np.array
        Flux densities in Jy, shape (n_nodes, n_sources)

    Returns
    -------
    coeffs : np.array
        Shape (n_nodes, n_sources), highest degree first, polynomial in
        ln(nu_GHz) giving the flux density in Jy directly (PySM
        ``logpolycoefflux`` convention)
    """
    x = np.log(np.asarray(freqs_ghz, dtype=np.float64))
    n = len(x)
    vander = np.empty((n, n))
    for i_row, x_row in enumerate(x):
        for i_coeff in range(n):
            vander[i_row, i_coeff] = x_row ** (n - 1 - i_coeff)
    return np.linalg.inv(vander) @ flux


def read_columns(filename, columns):
    """Read FITS binary table columns into a dictionary of numpy arrays."""
    with fits.open(filename, memmap=True) as hdul:
        return {c: np.asarray(hdul[1].data[c]) for c in columns}


def evaluate_poly(coeffs, freq_ghz):
    """Evaluate the flux-density polynomial at one frequency for all sources."""
    x = np.log(freq_ghz)
    n = coeffs.shape[0]
    out = np.zeros(coeffs.shape[1])
    for i_coeff in range(n):
        out += coeffs[i_coeff] * x ** (n - 1 - i_coeff)
    return np.maximum(out, 0)


def convert(input_filename, output_filename, lensed, cutoff_mjy, ref_freq_ghz):
    """Convert an Agora radio catalog to the PointSourceCatalog HDF5 format."""
    if lensed:
        ra_col, dec_col = "RAL", "DECL"
    else:
        ra_col, dec_col = "RA", "DEC"
    columns = (
        [ra_col, dec_col, "psi"]
        + [f"I{int(f)}" for f in INTENSITY_FREQS_GHZ]
        + [
            c + str(int(f))
            for f in POLARIZED_FREQS_GHZ
            for c in ["Q", "U"]
        ]
    )
    print(f"Reading {input_filename}")
    data = read_columns(input_filename, columns)
    n_sources = len(data[ra_col])
    print(f"Sources in input catalog: {n_sources:,}")

    flux_i = np.vstack(
        [data[f"I{int(f)}"].astype(np.float64) for f in INTENSITY_FREQS_GHZ]
    )
    flux_p = np.vstack(
        [
            np.hypot(
                data[f"Q{int(f)}"].astype(np.float64),
                data[f"U{int(f)}"].astype(np.float64),
            )
            for f in POLARIZED_FREQS_GHZ
        ]
    )

    keep = np.ones(n_sources, dtype=bool)
    if cutoff_mjy > 0:
        ref_flux = data[f"I{int(ref_freq_ghz)}"].astype(np.float64)
        keep = ref_flux > cutoff_mjy * 1e-3
        print(
            f"Flux cutoff {cutoff_mjy} mJy at {ref_freq_ghz} GHz: "
            f"keeping {keep.sum():,} sources"
        )

    ra = data[ra_col].astype(np.float64)[keep]
    ra[ra < 0] += 360
    dec = data[dec_col].astype(np.float64)[keep]
    theta = np.radians(90 - dec)
    phi = np.radians(ra)

    coeff_i = poly_coeffs_through_points(
        INTENSITY_FREQS_GHZ, flux_i[:, keep]
    )
    coeff_p = poly_coeffs_through_points(
        POLARIZED_FREQS_GHZ, flux_p[:, keep]
    )
    psi = data["psi"].astype(np.float64)[keep]

    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        git_commit = "unknown"
    description = (
        "Agora radio source catalog (Omori 2024, arXiv:2212.07420) converted "
        "to the PySM PointSourceCatalog format for galsci/pysm issue #250. "
        "Flux model: polynomial in ln(nu_GHz) giving the flux density in Jy "
        "directly, intensity anchored at 5/95/150/220 GHz, polarized flux "
        "at 95/150/220 GHz. The dataset psi holds the per-source "
        "polarization angle in radians, not used by the current "
        "PointSourceCatalog implementation, kept for a future extension."
    )

    with h5py.File(output_filename, "w") as f:
        for name, array, units in [
            ("theta", theta, "rad"),
            ("phi", phi, "rad"),
            ("logpolycoefflux", coeff_i, "Jy"),
            ("logpolycoefpolflux", coeff_p, "Jy"),
            ("psi", psi, "rad"),
        ]:
            dset = f.create_dataset(name, data=array.astype(np.float64))
            dset.attrs["units"] = np.bytes_(units)
        f.attrs["description"] = np.bytes_(description)
        f.attrs["reference_frequency_GHz"] = ref_freq_ghz
        f.attrs["flux_cutoff_mJy"] = cutoff_mjy
        f.attrs["polynomial_degree"] = len(INTENSITY_FREQS_GHZ) - 1
        f.attrs["sorted_by"] = np.bytes_("input catalog order")
        f.attrs["ref_frame"] = np.bytes_("Equatorial")
        f.attrs["lensed"] = lensed
        f.attrs["generated_utc"] = np.bytes_(
            datetime.now(timezone.utc).isoformat()
        )
        f.attrs["git_commit"] = np.bytes_(git_commit)
        f.attrs["command"] = np.bytes_(command)
        f.attrs["input_catalog"] = np.bytes_(Path(input_filename).name)
    print(f"Written {output_filename}")


def validate(output_filename, n_samples, seed):
    """Check that the stored coefficients reproduce the catalog fluxes.

    The polynomial is exact at the fit nodes by construction; this verifies
    the round trip through the HDF5 file (storage dtype, coefficient
    ordering) on a random subsample.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(output_filename) as f:
        n_sources = f["logpolycoefflux"].shape[1]
        sub = np.sort(
            rng.choice(n_sources, min(n_samples, n_sources), replace=False)
        )
        coeff_i = np.array(f["logpolycoefflux"][:, sub])
        coeff_p = np.array(f["logpolycoefpolflux"][:, sub])
    for coeffs, freqs, label in [
        (coeff_i, INTENSITY_FREQS_GHZ, "I"),
        (coeff_p, POLARIZED_FREQS_GHZ, "P"),
    ]:
        worst = 0
        for freq in freqs:
            recovered = evaluate_poly(coeffs, freq)
            worst = max(worst, float(np.min(recovered)))
        print(
            f"{label}: max recovered flux at the nodes {worst:.3e} Jy "
            f"({len(sub):,} sources sampled, all finite: "
            f"{np.isfinite(coeff_i).all()})"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="See the module docstring for the full documentation.",
    )
    parser.add_argument(
        "--input", required=True, help="Agora radio catalog FITS filename"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HDF5 filename in PointSourceCatalog format",
    )
    parser.add_argument(
        "--lensed",
        required=True,
        choices=["true", "false"],
        help="Whether the input catalog is the lensed (len) or unlensed "
        "(unl) variant, this selects the position columns",
    )
    parser.add_argument(
        "--cutoff-mjy",
        type=float,
        default=0,
        help="Optional flux density cutoff at the reference frequency, "
        "keep all sources when 0 (default)",
    )
    parser.add_argument(
        "--ref-freq",
        type=float,
        default=REFERENCE_FREQ_GHZ,
        help="Reference frequency in GHz for the flux cutoff",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=10000,
        help="Number of sources sampled by the output validation",
    )
    args = parser.parse_args(argv)
    convert(
        args.input,
        args.output,
        args.lensed == "true",
        args.cutoff_mjy,
        args.ref_freq,
    )
    validate(args.output, args.validate_samples, seed=20261005)
    return 0


if __name__ == "__main__":
    sys.exit(main())
