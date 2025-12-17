"""Command-line tool for bandpass resampling.

This tool reads bandpass data from ASCII files and generates resampled
bandpasses using kernel density estimation. Output is saved in IPAC ASCII
table format compatible with astropy.

Examples
--------
Resample a single bandpass into 4 wafer variations:
    $ pysm_bandpass_sampler input_bandpass.txt --num-wafers 4 --output-dir ./output

Specify bootstrap size and random seed:
    $ pysm_bandpass_sampler input.txt -n 8 -b 256 --seed 42 -o ./wafers
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

try:
    import pysm3
except ImportError:
    print("Error: pysm3 is not installed. Install with: pip install pysm3")
    sys.exit(1)


def load_bandpass(filename):
    """Load bandpass data from an ASCII file.

    Parameters
    ----------
    filename : str or Path
        Path to ASCII file with two columns: frequency (GHz) and weights

    Returns
    -------
    nu : ndarray
        Frequencies in GHz
    bnu : ndarray
        Bandpass weights
    """
    try:
        data = np.loadtxt(filename, skiprows=1)
        if data.shape[1] < 2:
            raise ValueError("Input file must have at least 2 columns")
        nu = data[:, 0]
        bnu = data[:, 1]
        return nu, bnu
    except Exception as e:
        print(f"Error loading bandpass from {filename}: {e}")
        sys.exit(1)


def save_resampled_bandpass(result, output_path):
    """Save a resampled bandpass to IPAC table format.

    Parameters
    ----------
    result : dict
        Dictionary with 'frequency' and 'weights' arrays
    output_path : str or Path
        Output file path
    """
    table = Table()
    table["bandpass_frequency"] = result["frequency"] * u.GHz
    table["bandpass_weights"] = result["weights"]

    table.meta["comments"] = [
        f"Resampled bandpass centroid: {result['centroid']:.4f} GHz",
        f"Resampled bandpass bandwidth: {result['bandwidth']:.4f} GHz",
    ]

    table.write(output_path, format="ascii.ipac", overwrite=True)


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Resample bandpasses using kernel density estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input bandpass file (ASCII format with frequency and weights columns)",
    )

    parser.add_argument(
        "-n",
        "--num-wafers",
        type=int,
        default=1,
        help="Number of resampled bandpasses to generate (default: 1)",
    )

    parser.add_argument(
        "-b",
        "--bootstrap-size",
        type=int,
        default=128,
        help="Number of bootstrap samples (default: 128). Larger values give smoother results.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for resampled bandpasses (default: current directory)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="resampled_bandpass",
        help="Prefix for output filenames (default: resampled_bandpass)",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsample input bandpass by taking every Nth point (default: None, use all points)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input bandpass
    print(f"Loading bandpass from {args.input}...")
    nu, bnu = load_bandpass(args.input)

    # Optionally downsample
    if args.downsample:
        nu = nu[:: args.downsample]
        bnu = bnu[:: args.downsample]
        print(f"Downsampled to {len(nu)} points (every {args.downsample}th point)")

    print(f"Input bandpass: {len(nu)} frequency points")

    # Compute original bandpass moments
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    bnu_norm = bnu / trapezoid(bnu, nu)
    orig_centroid, orig_bandwidth = pysm3.compute_moments(nu, bnu_norm)
    print(f"Original centroid: {orig_centroid:.4f} GHz")
    print(f"Original bandwidth: {orig_bandwidth:.4f} GHz")

    # Resample bandpass
    print(
        f"\nResampling with {args.bootstrap_size} bootstrap samples "
        f"for {args.num_wafers} wafer(s)..."
    )
    try:
        results = pysm3.resample_bandpass(
            nu,
            bnu,
            num_wafers=args.num_wafers,
            bootstrap_size=args.bootstrap_size,
            random_seed=args.seed,
        )
    except ImportError as e:
        print(f"\nError: {e}")
        print(
            "\nTo use bandpass resampling, install scikit-learn:"
            "\n  pip install scikit-learn"
        )
        sys.exit(1)

    print("\nResampled bandpasses:")
    print(f"{'Wafer':<8} {'Centroid (GHz)':<18} {'Bandwidth (GHz)':<18}")
    print("-" * 44)

    # Save results
    for i, result in enumerate(results):
        print(f"{i:<8} {result['centroid']:>16.4f}  {result['bandwidth']:>16.4f}")

        # Generate output filename
        input_stem = Path(args.input).stem
        output_filename = f"{args.prefix}_{input_stem}_w{i}.tbl"
        output_path = output_dir / output_filename

        save_resampled_bandpass(result, output_path)
        print(f"         -> {output_path}")

    print(f"\nSuccessfully saved {len(results)} resampled bandpass(es)")


if __name__ == "__main__":
    main()
