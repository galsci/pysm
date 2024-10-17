#!/bin/bash
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --time=6:30:00
#SBATCH --cpus-per-task=48
#SBATCH --array=0-13

echo $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=48

export PYTHONUNBUFFERED=1

for nside in 2048 4096 8192
do
python create_catalog_background.py /mnt/sdceph/users/azonca/pysm-data/websky/0.4/radio_catalog/background/websky_full_catalog_trasp.h5 $nside
done
