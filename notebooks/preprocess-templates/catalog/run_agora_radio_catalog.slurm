#!/bin/bash
#SBATCH --job-name=agora_radio_cat
#SBATCH --account=mp107d
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=agora_radio_cat_%j.out
#SBATCH --error=agora_radio_cat_%j.err

set -euo pipefail

echo "Started: $(date -Is)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

AGORA_RADIO_DATA=${AGORA_RADIO_DATA:-$HOME/agora_radio_data}
PYTHON=${PYTHON:-$HOME/venvs/agora_radio/bin/python}

srun "$PYTHON" agora_radio_sources_catalog.py \
    --input "$AGORA_RADIO_DATA/catalogs/agora_radiocat_unl_universemachine_trinity_95_150_220ghz_randflux_datta2018_truncgauss.0.fits" \
    --output "$AGORA_RADIO_DATA/converted/agora_radio_unl_pysm.h5" \
    --lensed false

srun "$PYTHON" agora_radio_sources_catalog.py \
    --input "$AGORA_RADIO_DATA/catalogs/agora_radiocat_len_universemachine_trinity_95_150_220ghz_randflux_datta2018_truncgauss.0.fits" \
    --output "$AGORA_RADIO_DATA/converted/agora_radio_len_pysm.h5" \
    --lensed true

echo "Finished: $(date -Is)"
