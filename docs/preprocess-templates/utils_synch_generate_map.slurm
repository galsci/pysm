#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl
#SBATCH --error=synch_generate_map_%j.err
#SBATCH --output=synch_generate_map_%j.out
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=68
export HDF5_USE_FILE_LOCKING=FALSE

PAPERMILL="/global/common/software/cmb/zonca/conda/pycmb/bin/papermill"
for NSIDE in 4096 8192
do
    for NB in "utils_synch_generate_map.ipynb"
    do
        srun --cpu_bind=cores $PAPERMILL $NB data/${NB/.ipynb/.$NSIDE.$NAME.ipynb} -p output_nside $NSIDE -p name $NAME
    done
done
