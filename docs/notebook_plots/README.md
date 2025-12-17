# Bandpass Sampling Notebook Output Plots

These plots are generated from executing the `docs/bandpass_sampling.ipynb` notebook.

## Plot 1: Synthetic Input Bandpass (`01_synthetic_bandpass.png`)
Shows the test bandpass used for demonstration:
- Primary Gaussian at 100 GHz with 8 GHz width
- Secondary lobe at 110 GHz (10% amplitude)
- Added noise for realism
- **Statistics**: Centroid = 100.40 GHz, Bandwidth = 7.73 GHz

## Plot 2: Resampled Bandpasses (`02_resampled_bandpasses.png`)
Six resampled bandpasses representing different detector wafers:
- Black dashed line: Original bandpass
- Orange solid line: Resampled version
- Each subplot shows a different wafer with its centroid and bandwidth
- Demonstrates realistic detector-to-detector variations

## Plot 3: Statistical Distributions (`03_moment_distributions.png`)
Distribution analysis from 50 resampled bandpasses:
- **Left panel**: Centroid distribution (mean=100.58 GHz, σ=0.55 GHz)
- **Right panel**: Bandwidth distribution (mean=8.00 GHz, σ=0.39 GHz)
- Red lines: Original values
- Green lines: Mean of resampled distributions
- Shows the resampling process preserves statistical properties

## Plot 4: Overlay (`04_overlay.png`)
Overlay of 10 resampled bandpasses on the original:
- Thick black line: Original bandpass
- Semi-transparent orange lines: 10 resampled versions
- Demonstrates consistency while showing realistic variability
- All bandpasses are smooth and artifact-free from KDE

## Purpose
These plots demonstrate that the bandpass sampling implementation:
1. Correctly implements the MBS 16 approach
2. Produces realistic variations suitable for systematic studies
3. Preserves statistical properties of the original bandpass
4. Generates smooth, high-quality resampled bandpasses

---
*Note: These plots are temporarily added to the repository for review purposes and will be removed after PR merge.*
