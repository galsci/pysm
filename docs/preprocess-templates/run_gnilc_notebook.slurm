#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl
#SBATCH --cpus-per-task=272
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

PAPERMILL="/global/homes/z/zonca/condanamaster2/bin/papermill"
NB="expansion_dust_pysm3.ipynb"

NSIDE="2048_reg_queue_rerun_1.15"
srun --cpu_bind=cores $PAPERMILL $NB data/${NB/.ipynb/_$NSIDE.ipynb}
