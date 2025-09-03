#!/bin/bash
### Perlmutter
###SBATCH --qos=regular
###SBATCH --constraint=cpu
### Popeye
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Executing with $OMP_NUM_THREADS threads

export PYTHONUNBUFFERED=1

python websky_sources_high_flux_catalog_out_1mJy.py
