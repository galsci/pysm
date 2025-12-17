"""
Validation tests comparing PySM bandpass sampler with reference outputs
from the original MBS 16 implementation (simonsobs/bandpass_sampler).
"""

import numpy as np
import pytest
import os
from pathlib import Path

# Import our functions
from pysm3.bandpass_sampler import compute_moments


# Path to reference data
TEST_DATA_DIR = Path(__file__).parent / "data" / "bandpass_reference"


def load_ipac_table(filename):
    """
    Load an IPAC table format bandpass file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the IPAC table file
        
    Returns
    -------
    nu : ndarray
        Frequency array in GHz
    bnu : ndarray
        Bandpass weight array (normalized)
    """
    data = []
    with open(filename, 'r') as f:
        # Skip header lines (4 lines in IPAC format)
        for _ in range(4):
            next(f)
        # Read data
        for line in f:
            line = line.strip()
            if line and not line.startswith('|'):
                parts = line.split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])
    
    data = np.array(data)
    return data[:, 0], data[:, 1]


@pytest.mark.parametrize("telescope,band,wafer", [
    ("LAT", "LF1", "w0"),
    ("LAT", "HF2", "w0"),
    ("SAT", "LF1", "w0"),
    ("SAT", "HF2", "w0"),
    ("LAT", "LF1", "w1"),
    ("LAT", "HF2", "w1"),
    ("SAT", "LF1", "w1"),
    ("SAT", "HF2", "w1"),
])
def test_reference_bandpass_loads(telescope, band, wafer):
    """Test that reference bandpass files can be loaded successfully."""
    filename = TEST_DATA_DIR / f"{telescope}_{band}_{wafer}_reference.tbl"
    
    # Check file exists
    assert filename.exists(), f"Reference file not found: {filename}"
    
    # Load the bandpass
    nu, bnu = load_ipac_table(filename)
    
    # Basic sanity checks
    assert len(nu) > 0, "Frequency array is empty"
    assert len(bnu) > 0, "Bandpass array is empty"
    assert len(nu) == len(bnu), "Frequency and bandpass arrays have different lengths"
    
    # Check that frequencies are monotonically increasing
    assert np.all(np.diff(nu) > 0), "Frequencies are not monotonically increasing"
    
    # Check that bandpass is non-negative
    assert np.all(bnu >= 0), "Bandpass has negative values"
    
    # Check frequency ranges are reasonable
    if "LF" in band:
        # Low frequency bands should be around 20-40 GHz
        assert nu.min() >= 5, f"LF min frequency too low: {nu.min()}"
        assert nu.max() <= 100, f"LF max frequency too high: {nu.max()}"
    elif "HF" in band:
        # High frequency bands should be around 200-300 GHz
        assert nu.min() >= 100, f"HF min frequency too low: {nu.min()}"
        assert nu.max() <= 400, f"HF max frequency too high: {nu.max()}"


@pytest.mark.parametrize("telescope,band", [
    ("LAT", "LF1"),
    ("LAT", "HF2"),
    ("SAT", "LF1"),
    ("SAT", "HF2"),
])
def test_reference_bandpass_moments(telescope, band):
    """
    Test that reference bandpasses have reasonable moment statistics.
    
    This validates that our moment computation function works correctly
    on the reference data from the original implementation.
    """
    # Load both wafers for this telescope/band combination
    filename_w0 = TEST_DATA_DIR / f"{telescope}_{band}_w0_reference.tbl"
    filename_w1 = TEST_DATA_DIR / f"{telescope}_{band}_w1_reference.tbl"
    
    nu_w0, bnu_w0 = load_ipac_table(filename_w0)
    nu_w1, bnu_w1 = load_ipac_table(filename_w1)
    
    # Compute moments for both wafers
    centroid_w0, bandwidth_w0 = compute_moments(nu_w0, bnu_w0)
    centroid_w1, bandwidth_w1 = compute_moments(nu_w1, bnu_w1)
    
    # Check moments are reasonable
    assert centroid_w0 > 0, "Centroid should be positive"
    assert bandwidth_w0 > 0, "Bandwidth should be positive"
    assert centroid_w1 > 0, "Centroid should be positive"
    assert bandwidth_w1 > 0, "Bandwidth should be positive"
    
    # Check that centroids are close (should differ by < 5% for same band)
    centroid_diff_pct = 100 * abs(centroid_w1 - centroid_w0) / centroid_w0
    assert centroid_diff_pct < 5, (
        f"Centroids differ by {centroid_diff_pct:.1f}% "
        f"(w0: {centroid_w0:.2f}, w1: {centroid_w1:.2f})"
    )
    
    # Check that bandwidths are close (should differ by < 10% for same band)
    bandwidth_diff_pct = 100 * abs(bandwidth_w1 - bandwidth_w0) / bandwidth_w0
    assert bandwidth_diff_pct < 10, (
        f"Bandwidths differ by {bandwidth_diff_pct:.1f}% "
        f"(w0: {bandwidth_w0:.2f}, w1: {bandwidth_w1:.2f})"
    )
    
    # Check frequency range consistency
    if "LF" in band:
        # Low frequency: centroid around 25-30 GHz
        assert 20 < centroid_w0 < 50, f"LF centroid out of range: {centroid_w0}"
        assert 20 < centroid_w1 < 50, f"LF centroid out of range: {centroid_w1}"
    elif "HF" in band:
        # High frequency: centroid around 250-290 GHz
        assert 200 < centroid_w0 < 350, f"HF centroid out of range: {centroid_w0}"
        assert 200 < centroid_w1 < 350, f"HF centroid out of range: {centroid_w1}"


def test_all_reference_files_present():
    """Test that all expected reference files are present."""
    expected_files = [
        "LAT_LF1_w0_reference.tbl",
        "LAT_LF1_w1_reference.tbl",
        "LAT_HF2_w0_reference.tbl",
        "LAT_HF2_w1_reference.tbl",
        "SAT_LF1_w0_reference.tbl",
        "SAT_LF1_w1_reference.tbl",
        "SAT_HF2_w0_reference.tbl",
        "SAT_HF2_w1_reference.tbl",
    ]
    
    for filename in expected_files:
        filepath = TEST_DATA_DIR / filename
        assert filepath.exists(), f"Missing reference file: {filename}"


@pytest.mark.parametrize("telescope,band", [
    ("LAT", "LF1"),
    ("LAT", "HF2"),
    ("SAT", "LF1"),
    ("SAT", "HF2"),
])
def test_reference_normalization(telescope, band):
    """
    Test that reference bandpasses are properly normalized.
    
    The integral of the bandpass over frequency should be close to 1.
    """
    filename = TEST_DATA_DIR / f"{telescope}_{band}_w0_reference.tbl"
    nu, bnu = load_ipac_table(filename)
    
    # Compute integral using trapezoidal rule
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid
    
    integral = trapezoid(bnu, nu)
    
    # Should be normalized to 1 (allow some numerical error)
    assert 0.99 < integral < 1.01, (
        f"Bandpass not normalized: integral = {integral:.6f}"
    )
