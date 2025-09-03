import sys
import runpy
from pathlib import Path

import numpy as np
import xarray as xr
import h5py


def create_catalogs(base_flux_100, freqs, out_dir: Path):
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, np.pi, len(base_flux_100))
    phi = rng.uniform(0, 2 * np.pi, len(base_flux_100))
    for f in freqs:
        # simple power law spectrum per source with small per-source spectral index variation
        alpha = -0.7 + 0.1 * rng.normal(size=len(base_flux_100))
        flux = base_flux_100 * (f / 100.0) ** alpha
        pol = 0.1 * flux  # 10% polarization
        with h5py.File(out_dir / f"catalog_{f:.1f}.h5", "w") as h:
            h.create_dataset("flux", data=flux)
            h.create_dataset("polarized flux", data=pol)
            h.create_dataset("theta", data=theta)
            h.create_dataset("phi", data=phi)


def test_high_flux_catalog_sorting(tmp_path: Path):
    # Frequencies (keep small for speed)
    freqs = [70.0, 100.0, 143.0, 217.0]
    # Base fluxes at 100 GHz (Jy), intentionally unsorted
    base_flux_100 = np.array([
        0.0025,
        0.0012,
        0.0008,
        0.0030,
        0.0018,
        0.0005,
        0.0040,
        0.0022,
        0.0009,
        0.00105,
        0.0035,
        0.0007,
    ])
    cutoff_mjy = 1.0
    cutoff_jy = cutoff_mjy * 1e-3
    expected_selected = np.where(base_flux_100 > cutoff_jy)[0]
    expected_sorted = expected_selected[np.argsort(base_flux_100[expected_selected])[::-1]]

    create_catalogs(base_flux_100, freqs, tmp_path)

    output_file = tmp_path / "out.h5"
    # Assume script is in the same directory as this test file.
    script_path = Path(__file__).parent / "websky_sources_high_flux_catalog_out_1mJy.py"
    assert script_path.exists(), "Processing script not found"

    argv_backup = sys.argv.copy()
    try:
        sys.argv = [
            str(script_path),
            "--input-pattern",
            str(tmp_path / "catalog_{freq:.1f}.h5"),
            "--frequencies",
            *map(str, freqs),
            "--cutoff-mjy",
            str(cutoff_mjy),
            "--ref-freq",
            "100.0",
            "--degree",
            "4",
            "--progress-every",
            "1000",
            "--output",
            str(output_file),
        ]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = argv_backup

    ds = xr.load_dataset(output_file)

    # Assert indices match expected selected sources (same set)
    produced_indices = ds.index.values
    assert set(produced_indices.tolist()) == set(expected_selected.tolist())

    # Evaluate polynomial at log(100 GHz) via numpy.polyval and assert descending order
    log_ref = np.log(100.0)
    coeffs = ds.logpolycoefflux.values  # shape (deg+1, Nsel) highest degree first along power axis
    fitted = np.polyval(coeffs, log_ref)
    # Check sorting (non-increasing)
    assert np.all(fitted[:-1] >= fitted[1:] - 1e-12)

    # Check ordering corresponds to base flux ordering (allow small tolerance)
    base_flux_selected = base_flux_100[produced_indices]
    assert np.all(base_flux_selected[:-1] >= base_flux_selected[1:] - 1e-6)

    # Metadata checks
    assert "sorted_by" in ds.attrs
    assert "polynomial_degree" in ds.attrs and ds.attrs["polynomial_degree"] == 4
