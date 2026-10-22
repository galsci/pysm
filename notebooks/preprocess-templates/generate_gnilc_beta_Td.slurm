#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl
#SBATCH --cpus-per-task=272
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=68
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

PAPERMILL="/global/homes/z/zonca/condanamaster2/bin/papermill"
NB="utils_gnilc_generate_map_spectralindex_Td.ipynb"
for NSIDE in 2048 4096 8192
do
    for NAME in "beta" "Td"
    do
        srun --cpu_bind=cores $PAPERMILL $NB data/${NB/.ipynb/.$NSIDE.$NAME.ipynb} -p output_nside $NSIDE -p name $NAME
    done
done
