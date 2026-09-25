#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --error=synch_low_freq_s8_%j.err
#SBATCH --output=synch_low_freq_s8_%j.out
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL

# Build the s8 low-frequency synchrotron templates at 1 degree (issue #274).
# The executed copy of the notebook is written to a separate output file,
# the source notebook in the repository is never modified.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

# The published templates must be readable by the portal.nersc.gov web server
umask 022

# Papermill from the CMB collaboration conda environment
PAPERMILL="/global/common/software/cmb/zonca/conda/pycmb/bin/papermill"

NB="synchrotron_low_freq_s8.ipynb"

# Publish the templates to the PySM data release folder, served at
# https://portal.nersc.gov/project/cmb/pysm-data/lowfreq/
OUTPUT_FOLDER=/global/cfs/cdirs/cmb/www/pysm-data/lowfreq
DOWNLOAD_FOLDER=$PWD/data

mkdir -p "$OUTPUT_FOLDER" "$DOWNLOAD_FOLDER"

srun --cpu_bind=cores $PAPERMILL \
    $NB \
    "data/${NB%.ipynb}_executed.$SLURM_JOB_ID.ipynb" \
    -p output_folder "$OUTPUT_FOLDER" \
    -p download_folder "$DOWNLOAD_FOLDER"
