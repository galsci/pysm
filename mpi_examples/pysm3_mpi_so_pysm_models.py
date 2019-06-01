import numpy as np
import pysm
import pysm.units as u
from so_pysm_models import get_so_models

from mpi4py import MPI

nside = 32

map_dist = pysm.MapDistribution(
    pixel_indices=None, nside=nside, mpi_comm=MPI.COMM_WORLD
)

components = []
for comp in ["SO_d0", "SO_s0", "SO_f0", "SO_a0"]:
    components.append(get_so_models(comp, nside, map_dist=map_dist))

sky = pysm.Sky(nside=nside, component_objects=components, map_dist=map_dist)

m = sky.get_emission(
    freq=np.arange(50, 55) * u.GHz, weights=np.array([0.1, 0.3, 0.5, 0.3, 0.1])
)[0]

print(map_dist.mpi_comm.rank, m.shape, m.min(), m.max())

m_smoothed = pysm.apply_smoothing_and_coord_transform(
    m, fwhm=1 * u.deg, map_dist=map_dist
)

print(map_dist.mpi_comm.rank, m_smoothed.shape, m_smoothed.min(), m_smoothed.max())
