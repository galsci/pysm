#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --time=1:30:00
#SBATCH --array=0-13

echo $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=128

export PYTHONUNBUFFERED=1

/global/common/software/cmb/zonca/conda/pycmb/bin/python create_catalog_background.py 4096
