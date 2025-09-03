#!/bin/bash
# SLURM job script to build the high-flux Websky catalog with explicit parameters.

set -euo pipefail

### Perlmutter (commented examples)
###SBATCH --qos=regular
###SBATCH --constraint=cpu
### Popeye
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=ALL

echo "Started: $(date -Is)"
echo "SLURM_NODELIST=$SLURM_NODELIST"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Frequencies explicitly listed (GHz)
FREQS="18.7 24.5 44.0 70.0 100.0 143.0 217.0 353.0 545.0 643.0 729.0 857.0 906.0"

# Run with all parameters explicit
python websky_sources_high_flux_catalog_out_1mJy.py \
	--input-pattern data/matched_catalogs_2/catalog_{freq:.1f}.h5 \
	--frequencies $FREQS \
	--cutoff-mjy 1.0 \
	--ref-freq 100.0 \
	--degree 4 \
	--progress-every 500 \
	--output data/websky_high_flux_catalog_1mJy_sorted_by_fit.h5

echo "Finished: $(date -Is)"
