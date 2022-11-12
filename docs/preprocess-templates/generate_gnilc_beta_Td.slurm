#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl

export OMP_NUM_THREADS=68

PAPERMILL="/global/common/software/cmb/zonca/conda/pycmb/bin/papermill"
NB="utils_gnilc_generate_map_spectralindex_Td.ipynb"
for NSIDE in 4096 8192
do
    for NAME in "beta" "Td"
    do
        $PAPERMILL $NB data/${NB/.ipynb/.$NSIDE.$NAME.ipynb} -p output_nside $NSIDE -p name $NAME
    done
done
