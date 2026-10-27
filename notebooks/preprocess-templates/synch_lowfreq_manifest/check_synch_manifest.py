#!/usr/bin/env python3
"""Verify the published PySM s8 templates against an immutable manifest.

Downloads (or reads locally) each file listed in the manifest and compares its
SHA-256 and size to the recorded values. Exits non-zero if any entry is missing,
corrupted, or has the wrong size, so it can be used as a nightly/weekly CI check
that the published data the ``[s8]`` preset relies on has not changed.

The check is intentionally stateless and uses only the standard library, so it
runs on a GitHub-hosted runner with access to the public data URL (no NERSC
credentials needed).

Usage
-----
    python check_synch_manifest.py                      # use the directory default
    python check_synch_manifest.py --manifest manifest_2026.10.22.txt
    python check_synch_manifest.py --local-data <dir>   # read from a local folder
"""

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = "manifest_2026.10.22.txt"
DATA_URL = "https://portal.nersc.gov/project/cmb/pysm-data/synch_lowfreq/"

ENTRY = re.compile(r"^\s*([0-9a-fA-F]{64})\s+(\d+)\s+(\S+)\s*$")


def parse_manifest(path):
    entries = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ENTRY.match(line)
        if not m:
            raise ValueError(f"{path}: malformed entry on line {lineno}: {line!r}")
        entries.append((m.group(3), m.group(2), m.group(1)))
    return entries


def read_file(local_dir, filename):
    """Return the local bytes if available, otherwise download from the data URL."""
    if local_dir is not None:
        candidate = Path(local_dir) / filename
        if candidate.exists():
            return candidate.read_bytes()
    url = DATA_URL + filename
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001 - surface any fetch problem as a failure
        raise FileNotFoundError(f"could not fetch {url}: {exc}")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        default=str(here / DEFAULT_MANIFEST),
        help=f"manifest file (default {here / DEFAULT_MANIFEST})",
    )
    ap.add_argument(
        "--local-data",
        default=None,
        help="optional local folder containing the files (fast path; avoids downloading)",
    )
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    entries = parse_manifest(manifest)
    print(f"Checking {len(entries)} templates against {manifest.name} "
          f"(data URL: {DATA_URL})")

    failed = 0
    missing = 0
    for filename, size, expected_sha in entries:
        try:
            data = read_file(args.local_data, filename)
        except FileNotFoundError as exc:
            print(f"  [MISSING] {filename}: {exc}")
            missing += 1
            failed += 1
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha.lower() != expected_sha.lower() or len(data) != int(size):
            print(f"  [FAIL]    {filename}: sha256 {sha[:16]}... size {len(data)} "
                  f"(expected {expected_sha[:16]}... / {size})")
            failed += 1
        else:
            print(f"  [ok]      {filename} ({len(data)} bytes)")

    print(f"{len(entries) - failed}/{len(entries)} matched, {missing} missing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
