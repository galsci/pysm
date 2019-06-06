import pysm

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
map_dist = pysm.MapDistribution(nside=64, mpi_comm=mpi_comm)

print(mpi_comm.rank)
pysm.models.template.read_alm("/home/zonca/s/simonsobs/so_pysm_models_data/websky/0.3/lensed_alm_seed1.fits", map_dist=map_dist)
