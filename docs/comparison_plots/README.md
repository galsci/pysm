# Bandpass Sampling Validation: Comparison with Reference Data

This directory contains validation plots comparing PySM's bandpass sampling implementation with reference outputs from the original MBS 16 (Map-Based Simulations, release 16) used by Simons Observatory.

## Reference Data Source

The reference bandpass files are from the [simonsobs/bandpass_sampler](https://github.com/simonsobs/bandpass_sampler) repository:
- **LAT** (Large Aperture Telescope): LF1 (low frequency) and HF2 (high frequency) bands
- **SAT** (Small Aperture Telescope): LF1 (low frequency) and HF2 (high frequency) bands
- **Wafers**: w0 and w1 for each telescope/band combination

## Plots

### 1. `comparison_reference_vs_resampled.png`

**Description:** Direct visual comparison of reference bandpasses (black lines) with PySM-generated resampled bandpasses (colored lines).

**What it shows:**
- **Black line**: Original reference bandpass from MBS 16
- **Colored lines (red, teal, mint)**: Three independent resampled bandpasses generated with different random seeds

**Key observations:**
- LAT LF1: Centroid ~26.15 GHz, Bandwidth ~2.75 GHz
- LAT HF2: Centroid ~286.33 GHz, Bandwidth ~16.86 GHz
- SAT LF1: Centroid ~26.12 GHz, Bandwidth ~2.76 GHz
- SAT HF2: Centroid ~283.48 GHz, Bandwidth ~16.00 GHz

All resampled bandpasses closely follow the reference shape while introducing realistic variations.

### 2. `comparison_statistical_distributions.png`

**Description:** Statistical validation showing the distribution of centroids from 10 independent resampling operations for each bandpass.

**What it shows:**
- **Histogram**: Distribution of centroid frequencies from 10 resampled bandpasses
- **Blue dashed line**: Reference centroid from original MBS 16 bandpass
- **Green solid line**: Mean centroid of resampled bandpasses

**Statistical results (N=10 samples):**

| Telescope-Band | Reference Centroid | Mean Resampled | Std Dev |
|----------------|-------------------|----------------|---------|
| LAT LF1        | 26.15 GHz         | 26.29 GHz      | 0.248 GHz |
| LAT HF2        | 286.33 GHz        | 287.17 GHz     | 1.490 GHz |
| SAT LF1        | 26.12 GHz         | 26.25 GHz      | 0.249 GHz |
| SAT HF2        | 283.48 GHz        | 284.26 GHz     | 1.414 GHz |

**Validation:**
- Resampled centroids are well-centered around reference values
- Variations are realistic (~1% of centroid frequency)
- Standard deviations are appropriate for detector-to-detector variability

## Conclusion

These plots validate that PySM's bandpass sampling implementation:

✅ **Correctly processes real SO bandpass data** in IPAC table format

✅ **Preserves bandpass shape** while introducing realistic variations

✅ **Maintains statistical consistency** with reference centroids and bandwidths

✅ **Produces physically realistic variations** suitable for large-scale simulations

The implementation is ready for use in map-based simulations and systematic studies requiring realistic detector bandpass variations.

## Regenerating Plots

To regenerate these comparison plots:

```bash
python3 scripts/generate_comparison_plots.py
```

This script:
1. Loads reference bandpasses from `tests/data/bandpass_reference/`
2. Generates resampled bandpasses using PySM's `resample_bandpass()` function
3. Compares statistical properties (centroids, bandwidths)
4. Saves plots to `docs/comparison_plots/`
