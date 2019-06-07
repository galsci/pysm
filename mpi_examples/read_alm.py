import pysm
import numpy as np
import healpy as hp

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
lmax = 16
map_dist = pysm.MapDistribution(nside=64, smoothing_lmax=lmax, mpi_comm=mpi_comm)


filename = "alm.fits"
alm_size = hp.Alm.getsize(lmax)
alm = np.arange(alm_size*3, dtype=np.complex64).reshape((3, alm_size))
alm += 1j * (np.arange(alm_size*3, dtype=np.complex64).reshape((3, alm_size)))/10
alm[:, :lmax] = alm[:, :lmax].real

hp.write_alm(filename, alm, overwrite=True)

print(mpi_comm.rank)
local_alm = pysm.models.template.read_alm(filename, map_dist=map_dist)

for i_pol in range(3):
    np.testing.assert_allclose(local_alm[0, i_pol, :lmax+1], i_pol*alm_size + np.arange(lmax+1))
    np.testing.assert_allclose(local_alm[0, i_pol, lmax+1::2], i_pol*alm_size + np.arange(lmax+1, alm_size))
    np.testing.assert_allclose(local_alm[0, i_pol, lmax+2::2], (i_pol*alm_size + np.arange(lmax+1, alm_size))/10)
