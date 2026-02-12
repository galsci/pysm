#!/usr/bin/env python
"""Precompute cached reference maps for gaussian foreground comparison notebook."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys
import time

import pysm3
import pysm3.units as u


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=64, help="Output nside for cached maps")
    parser.add_argument(
        "--dust-freq-ghz",
        type=float,
        default=145.0,
        help="Dust comparison frequency in GHz",
    )
    parser.add_argument(
        "--sync-freq-ghz",
        type=float,
        default=30.0,
        help="Synchrotron comparison frequency in GHz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("docs/.cache"),
        help="Directory for .npz cache files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cache files even if they already exist",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal mode: generate only one preset in this process",
    )
    parser.add_argument(
        "--preset",
        choices=["d10", "s5", "gd0", "gs0"],
        help="Preset to generate in --worker mode",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["d10", "s5"],
        help="Presets to cache in non-worker mode (default: d10 s5)",
    )
    return parser.parse_args()


def ensure_cached_map(preset_string, nside, freq, cache_dir, force=False):
    cache_name = pysm3.get_cache_filename(preset_string, nside, freq.to_value(u.GHz))
    cache_path = cache_dir / cache_name

    if cache_path.exists() and not force:
        print(f"[cache] exists: {cache_path}")
        return cache_path

    if cache_path.exists() and force:
        cache_path.unlink()

    t0 = time.time()
    pysm3.get_or_create_cached_preset_map(
        preset_string=preset_string,
        nside=nside,
        freq=freq,
        cache_dir=cache_dir,
    )
    dt = time.time() - t0
    print(f"[cache] generated: {cache_path} ({dt:.1f} s)")
    return cache_path


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    preset_freq = {
        "d10": args.dust_freq_ghz * u.GHz,
        "gd0": args.dust_freq_ghz * u.GHz,
        "s5": args.sync_freq_ghz * u.GHz,
        "gs0": args.sync_freq_ghz * u.GHz,
    }

    if args.worker:
        if args.preset is None:
            raise ValueError("--worker requires --preset")
        freq = preset_freq[args.preset]
        ensure_cached_map(args.preset, args.nside, freq, args.cache_dir, force=args.force)
        return

    # Run each heavy preset generation in its own process to avoid cumulative memory usage.
    for preset in args.presets:
        if preset not in preset_freq:
            raise ValueError(f"Unsupported preset: {preset}")
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--preset",
            preset,
            "--nside",
            str(args.nside),
            "--dust-freq-ghz",
            str(args.dust_freq_ghz),
            "--sync-freq-ghz",
            str(args.sync_freq_ghz),
            "--cache-dir",
            str(args.cache_dir),
        ]
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
