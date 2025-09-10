#!/bin/bash
#SBATCH --job-name=websky_cat_1mJy
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=websky_cat_%j.out
#SBATCH --error=websky_cat_%j.err
#SBATCH --mail-type=END,FAIL
## If you need email notifications specify: --mail-user=you@institution.edu

set -euo pipefail

echo "Started: $(date -Is)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Submit dir: $SLURM_SUBMIT_DIR"

# Ensure relative paths resolve from submit directory
cd "$SLURM_SUBMIT_DIR"

# Thread / buffering controls
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Frequencies explicitly listed (GHz)
FREQS="18.7 24.5 44.0 70.0 100.0 143.0 217.0 353.0 545.0 643.0 729.0 857.0 906.0"

# If a module or virtual env is needed, activate here, e.g.:
# module load python
# source /path/to/venv/bin/activate

# Run without --output to trigger auto-generated filename logic in the script
srun python websky_sources_high_flux_catalog_out_1mJy.py \
	--input-pattern data/matched_catalogs_2/catalog_{freq:.1f}.h5 \
	--frequencies $FREQS \
	--cutoff-mjy 1.0 \
	--ref-freq 100.0 \
	--degree 4 \
	--progress-every 500

echo "Finished: $(date -Is)"
