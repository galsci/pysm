#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --error=validate_synch_low_freq_s8_%j.err
#SBATCH --output=validate_synch_low_freq_s8_%j.out
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL

# Validate the s8 low-frequency synchrotron model (issue #274) against
# the s1 and s5 models, with all the checks of G. Puglisi's notebook.
# The executed copy of the notebook is written to a separate output file,
# the source notebook in the repository is never modified.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

# Papermill from the CMB collaboration conda environment
PAPERMILL="/global/common/software/cmb/zonca/conda/pycmb/bin/papermill"

NB="validate_synch_low_freq_s8.ipynb"

# The s8 templates and the s1/s5 inputs are read from the local PySM data
# release folder through PYSM_LOCAL_DATA (no download needed on NERSC).
export PYSM_LOCAL_DATA=/global/cfs/cdirs/cmb/www/pysm-data

mkdir -p data

srun --cpu_bind=cores $PAPERMILL \
    $NB \
    "data/${NB%.ipynb}_executed.$SLURM_JOB_ID.ipynb"
