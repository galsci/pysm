#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl

export OMP_NUM_THREADS=64
PAPERMILL="/global/common/software/cmb/zonca/conda/pycmb/bin/papermill"
NB="small_scale_dust_pysm3_generate_templates.ipynb"
for NSIDE in 4096 8192
do
        $PAPERMILL $NB data/${NB/.ipynb/_$NSIDE.ipynb} -p output_nside $NSIDE
done
