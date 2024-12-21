#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --time=2:00:00
#SBATCH --array=0-13

echo $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
echo Executing with $OMP_NUM_THREADS threads

export PYTHONUNBUFFERED=1

for nside in 2048 4096 8192
do
    python create_catalog_background.py $PYSM_LOCAL_DATA/websky/0.4/radio_catalog/background/websky_full_catalog_trasp.h5 $nside
done
