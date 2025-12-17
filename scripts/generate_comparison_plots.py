#!/usr/bin/env python
"""
Generate comparison plots between reference SO bandpasses and PySM resampled bandpasses.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pysm3

# Configuration
REFERENCE_DIR = Path(__file__).parent.parent / "tests/data/bandpass_reference"
OUTPUT_DIR = Path(__file__).parent.parent / "docs/comparison_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Files to compare
FILES = [
    "LAT_LF1_w0_reference.tbl",
    "LAT_HF2_w0_reference.tbl",
    "SAT_LF1_w0_reference.tbl",
    "SAT_HF2_w0_reference.tbl",
]

def load_ipac_table(filename):
    """Load IPAC table format bandpass file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (lines starting with | or \)
    data_lines = [line for line in lines if not line.startswith(('|', '\\'))]
    
    # Parse data
    freq = []
    transmission = []
    for line in data_lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq.append(float(parts[0]))
                    transmission.append(float(parts[1]))
                except ValueError:
                    continue
    
    return np.array(freq), np.array(transmission)

def resample_single_bandpass(nu, bnu, random_seed):
    """Generate a single resampled bandpass."""
    results = pysm3.resample_bandpass(nu, bnu, num_wafers=1, random_seed=random_seed)
    return results[0]['frequency'], results[0]['weights']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

print("Generating comparison plots...")

for idx, filename in enumerate(FILES):
    ax = axes[idx]
    filepath = REFERENCE_DIR / filename
    
    # Load reference bandpass
    nu_ref, bnu_ref = load_ipac_table(filepath)
    
    # Normalize reference bandpass
    integral = np.trapz(bnu_ref, nu_ref)
    bnu_ref_norm = bnu_ref / integral
    
    # Generate 3 resampled bandpasses with different seeds
    colors_resampled = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # Plot reference
    ax.plot(nu_ref, bnu_ref_norm, 'k-', linewidth=2.5, label='Reference (SO)', alpha=0.8)
    
    # Plot resampled versions
    for seed_offset, color in enumerate(colors_resampled):
        nu_new, bnu_new = resample_single_bandpass(nu_ref, bnu_ref_norm, 
                                                     random_seed=42 + seed_offset)
        ax.plot(nu_new, bnu_new, color=color, linewidth=1.5, alpha=0.7,
                label=f'Resampled #{seed_offset+1}')
    
    # Compute moments
    centroid_ref = pysm3.compute_moments(nu_ref, bnu_ref_norm)[0]
    bandwidth_ref = pysm3.compute_moments(nu_ref, bnu_ref_norm)[1]
    
    # Styling
    telescope = "LAT" if "LAT" in filename else "SAT"
    band = "LF1" if "LF1" in filename else "HF2"
    ax.set_title(f'{telescope} {band}\nCentroid: {centroid_ref:.2f} GHz, '
                 f'Bandwidth: {bandwidth_ref:.2f} GHz', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (GHz)', fontsize=10)
    ax.set_ylabel('Normalized Transmission', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='best')
    
    # Set reasonable axis limits
    ax.set_xlim(nu_ref.min() * 0.95, nu_ref.max() * 1.05)
    ax.set_ylim(0, bnu_ref_norm.max() * 1.15)
    
    print(f"  {filename}: centroid={centroid_ref:.2f} GHz, bandwidth={bandwidth_ref:.2f} GHz")

plt.tight_layout()
output_file = OUTPUT_DIR / "comparison_reference_vs_resampled.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_file}")

# Create a second figure showing statistical comparison
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

print("\nGenerating statistical comparison plots...")

for idx, filename in enumerate(FILES):
    ax = axes2[idx]
    filepath = REFERENCE_DIR / filename
    
    # Load reference bandpass
    nu_ref, bnu_ref = load_ipac_table(filepath)
    integral = np.trapz(bnu_ref, nu_ref)
    bnu_ref_norm = bnu_ref / integral
    
    # Generate 10 resampled bandpasses (reduced for speed)
    n_samples = 10
    centroids = []
    bandwidths = []
    
    for seed in range(n_samples):
        nu_new, bnu_new = resample_single_bandpass(nu_ref, bnu_ref_norm, random_seed=seed + 100)
        c, b = pysm3.compute_moments(nu_new, bnu_new)
        centroids.append(c)
        bandwidths.append(b)
    
    centroids = np.array(centroids)
    bandwidths = np.array(bandwidths)
    
    # Compute reference moments
    centroid_ref, bandwidth_ref = pysm3.compute_moments(nu_ref, bnu_ref_norm)
    
    # Create histogram
    ax.hist(centroids, bins=20, alpha=0.6, color='#FF6B6B', label='Resampled Centroids', 
            edgecolor='black', linewidth=0.5)
    ax.axvline(centroid_ref, color='blue', linewidth=2.5, linestyle='--', 
               label=f'Reference: {centroid_ref:.2f} GHz')
    ax.axvline(np.mean(centroids), color='green', linewidth=2, linestyle='-', 
               label=f'Mean: {np.mean(centroids):.2f} GHz')
    
    # Styling
    telescope = "LAT" if "LAT" in filename else "SAT"
    band = "LF1" if "LF1" in filename else "HF2"
    ax.set_title(f'{telescope} {band} - Centroid Distribution\n'
                 f'σ = {np.std(centroids):.3f} GHz (N={n_samples})', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Centroid Frequency (GHz)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    print(f"  {filename}: mean={np.mean(centroids):.2f} GHz, std={np.std(centroids):.3f} GHz")

plt.tight_layout()
output_file2 = OUTPUT_DIR / "comparison_statistical_distributions.png"
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_file2}")

print("\n✅ Comparison plots generated successfully!")
