import numpy as np
from mpi4py import MPI

import pysm3 as pysm
import pysm3.units as u

nside = 32

map_dist = pysm.MapDistribution(
    pixel_indices=None, nside=nside, mpi_comm=MPI.COMM_WORLD
)

sky = pysm.Sky(nside=nside, preset_strings=["d1", "s1", "f1", "a1"], map_dist=map_dist)

m = sky.get_emission(
    freq=np.arange(50, 55) * u.GHz, weights=np.array([0.1, 0.3, 0.5, 0.3, 0.1])
)[0]

print(map_dist.mpi_comm.rank, m.shape, m.min(), m.max())

m_smoothed = pysm.apply_smoothing_and_coord_transform(
    m, fwhm=1 * u.deg, map_dist=map_dist
)

print(map_dist.mpi_comm.rank, m_smoothed.shape, m_smoothed.min(), m_smoothed.max())
