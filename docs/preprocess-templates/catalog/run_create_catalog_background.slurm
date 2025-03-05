#!/bin/bash
### Perlmutter
###SBATCH --qos=regular
###SBATCH --constraint=cpu
### Popeye
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=48
#SBATCH --array=0-15
#SBATCH --mail-user=andrea@andreazonca.com
#SBATCH --mail-type=ALL

nside=8192

echo "Task ID: "$SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Executing with $OMP_NUM_THREADS threads

export PYTHONUNBUFFERED=1

python create_catalog_background.py $PYSM_LOCAL_DATA/websky/0.4/radio_catalog/background/websky_full_catalog_trasp.h5 $nside
