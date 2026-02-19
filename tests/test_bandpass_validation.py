from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

import pysm3


TEST_DATA_DIR = Path(__file__).parent / "data" / "bandpass_reference"


def load_ipac_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t = Table.read(path, format="ascii.ipac")
    nu = np.asarray(t["bandpass_frequency"], dtype=float)
    # Reference tables use `bandpass_weight` (singular). Keep compatibility with
    # any future tables written by our CLI (`bandpass_weights`).
    if "bandpass_weight" in t.colnames:
        bnu = np.asarray(t["bandpass_weight"], dtype=float)
    else:
        bnu = np.asarray(t["bandpass_weights"], dtype=float)
    return nu, bnu


def test_all_reference_files_present():
    expected = [
        "LAT_LF1_w0_reference.tbl",
        "LAT_LF1_w1_reference.tbl",
        "LAT_HF2_w0_reference.tbl",
        "LAT_HF2_w1_reference.tbl",
        "SAT_LF1_w0_reference.tbl",
        "SAT_LF1_w1_reference.tbl",
        "SAT_HF2_w0_reference.tbl",
        "SAT_HF2_w1_reference.tbl",
        "README.md",
    ]
    for name in expected:
        assert (TEST_DATA_DIR / name).exists(), f"Missing reference file: {name}"


@pytest.mark.parametrize(
    "filename,kind",
    [
        ("LAT_LF1_w0_reference.tbl", "LF"),
        ("LAT_LF1_w1_reference.tbl", "LF"),
        ("SAT_LF1_w0_reference.tbl", "LF"),
        ("SAT_LF1_w1_reference.tbl", "LF"),
        ("LAT_HF2_w0_reference.tbl", "HF"),
        ("LAT_HF2_w1_reference.tbl", "HF"),
        ("SAT_HF2_w0_reference.tbl", "HF"),
        ("SAT_HF2_w1_reference.tbl", "HF"),
    ],
)
def test_reference_bandpass_sanity(filename: str, kind: str):
    nu, bnu = load_ipac_table(TEST_DATA_DIR / filename)

    assert nu.ndim == 1 and bnu.ndim == 1
    assert nu.size == bnu.size
    assert nu.size > 10

    assert np.all(np.isfinite(nu))
    assert np.all(np.isfinite(bnu))
    assert np.all(np.diff(nu) > 0), "Frequencies must be strictly increasing"
    assert np.all(bnu >= 0), "Bandpass weights must be non-negative"

    try:
        from numpy import trapezoid
    except ImportError:  # pragma: no cover
        from numpy import trapz as trapezoid

    integ = trapezoid(bnu, nu)
    assert integ == pytest.approx(1.0, abs=1e-3)

    if kind == "LF":
        assert 5.0 <= nu.min() <= 40.0
        assert 35.0 <= nu.max() <= 100.0
    else:
        assert 100.0 <= nu.min() <= 260.0
        assert 260.0 <= nu.max() <= 400.0


@pytest.mark.parametrize(
    "filename,centroid_range",
    [
        ("LAT_LF1_w0_reference.tbl", (20.0, 50.0)),
        ("LAT_HF2_w0_reference.tbl", (200.0, 350.0)),
        ("SAT_LF1_w0_reference.tbl", (20.0, 50.0)),
        ("SAT_HF2_w0_reference.tbl", (200.0, 350.0)),
    ],
)
def test_reference_moments_are_reasonable(filename: str, centroid_range: tuple[float, float]):
    nu, bnu = load_ipac_table(TEST_DATA_DIR / filename)
    centroid, bandwidth = pysm3.compute_moments(nu, bnu)
    assert centroid_range[0] < centroid < centroid_range[1]
    assert bandwidth > 0


@pytest.mark.parametrize(
    "telescope,band",
    [
        ("LAT", "LF1"),
        ("LAT", "HF2"),
        ("SAT", "LF1"),
        ("SAT", "HF2"),
    ],
)
def test_wafers_have_similar_moments(telescope: str, band: str):
    nu0, b0 = load_ipac_table(TEST_DATA_DIR / f"{telescope}_{band}_w0_reference.tbl")
    nu1, b1 = load_ipac_table(TEST_DATA_DIR / f"{telescope}_{band}_w1_reference.tbl")
    c0, bw0 = pysm3.compute_moments(nu0, b0)
    c1, bw1 = pysm3.compute_moments(nu1, b1)

    assert abs(c1 - c0) / c0 < 0.05
    assert abs(bw1 - bw0) / bw0 < 0.10


@pytest.mark.parametrize(
    "filename",
    [
        "LAT_LF1_w0_reference.tbl",
        "LAT_HF2_w0_reference.tbl",
        "SAT_LF1_w0_reference.tbl",
        "SAT_HF2_w0_reference.tbl",
    ],
)
def test_resampling_preserves_centroid_scale(filename: str):
    nu, bnu = load_ipac_table(TEST_DATA_DIR / filename)
    c_ref, _ = pysm3.compute_moments(nu, bnu)
    out = pysm3.resample_bandpass(
        nu,
        bnu,
        num_wafers=1,
        bootstrap_size=64,
        random_seed=0,
    )[0]

    c_new = out["centroid"]
    assert abs(c_new - c_ref) / c_ref < 0.05
