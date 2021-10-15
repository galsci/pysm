#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=knl

export OMP_NUM_THREADS=64

/global/u2/z/zonca/condanamaster/bin/python generate_lower_nside_templates.py data/dust_gnilc_hybrid_out_alm_nside8192_lmax16384.fits
