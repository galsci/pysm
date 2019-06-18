#!/bin/bash
#SBATCH --partition=soap
#SBATCH --time=2:00:00
#SBATCH --nodes=10
#SBATCH --job-name=pysm3

export OMP_NUM_THREADS=48
export MLX5_SINGLE_THREADED=0
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
time srun -n 10 --cpus-per-task=$OMP_NUM_THREADS --mpi=pmi2 python pysm3_mpi_so_pysm_models.py &> pysm3_run_10_nodes.log
#time srun -n 40 --cpus-per-task=$OMP_NUM_THREADS --mpi=pmi2 python pysm3_mpi_so_pysm_models.py &> pysm3_run_10_nodes_4-MPI_12-OMP.log
