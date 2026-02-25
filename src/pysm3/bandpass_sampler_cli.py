"""Command-line tool for bandpass resampling.

Reads a 2-column ASCII file (frequency [GHz], weights) and writes one or more
resampled bandpasses as IPAC ASCII tables.

Example
-------
`pysm_bandpass_sampler input_bandpass.txt --num-wafers 4 --output-dir ./output`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

import pysm3


def load_bandpass(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(filename, ndmin=2)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"failed to read bandpass file {filename}: {e}") from e

    if data.shape[1] < 2:
        raise ValueError("input file must have at least 2 columns: frequency, weights")
    return data[:, 0], data[:, 1]


def save_resampled_bandpass(result: dict, output_path: str | Path) -> None:
    table = Table()
    table["bandpass_frequency"] = result["frequency"] * u.GHz
    table["bandpass_weights"] = result["weights"]
    table.meta["comments"] = [
        f"Resampled bandpass centroid: {result['centroid']:.6f} GHz",
        f"Resampled bandpass bandwidth: {result['bandwidth']:.6f} GHz",
    ]
    table.write(output_path, format="ascii.ipac", overwrite=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resample bandpasses with KDE")
    parser.add_argument("input", help="ASCII file with frequency[GHz] and weights")
    parser.add_argument("-n", "--num-wafers", type=int, default=1)
    parser.add_argument("-b", "--bootstrap-size", type=int, default=128)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-o", "--output-dir", default=".")
    parser.add_argument("--prefix", default="resampled_bandpass")
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Take every Nth input point before sampling",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nu, bnu = load_bandpass(args.input)
    if args.downsample:
        nu = nu[:: args.downsample]
        bnu = bnu[:: args.downsample]

    try:
        from numpy import trapezoid
    except ImportError:  # pragma: no cover
        from numpy import trapz as trapezoid

    bnu_norm = bnu / trapezoid(bnu, nu)
    centroid, bandwidth = pysm3.compute_moments(nu, bnu_norm)
    print(f"Input centroid: {centroid:.6f} GHz")
    print(f"Input bandwidth: {bandwidth:.6f} GHz")

    results = pysm3.resample_bandpass(
        nu,
        bnu,
        num_wafers=args.num_wafers,
        bootstrap_size=args.bootstrap_size,
        random_seed=args.seed,
    )

    for i, result in enumerate(results):
        input_stem = Path(args.input).stem
        out = output_dir / f"{args.prefix}_{input_stem}_w{i}.tbl"
        save_resampled_bandpass(result, out)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

