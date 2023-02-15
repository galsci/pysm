#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=272
#SBATCH --constraint=knl
#SBATCH --mail-user=zonca@sdsc.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=gnilc_gen_temp
#SBATCH --array=4096,8192

export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

PAPERMILL="/global/homes/z/zonca/condanamaster2/bin/papermill --no-progress-bar"
NB="small_scale_dust_pysm3_generate_templates.ipynb"
JUPYTER_KERNEL="condanamaster2"

NSIDE=$SLURM_ARRAY_TASK_ID
srun -n 1 -c 272 --cpu_bind=cores $PAPERMILL --kernel=$JUPYTER_KERNEL $NB data/${NB/.ipynb/_$NSIDE.ipynb} -p output_nside $NSIDE
