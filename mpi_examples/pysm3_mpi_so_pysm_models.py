import numpy as np
import healpy as hp
import pysm
import pysm.units as u
from so_pysm_models import get_so_models
import so_pysm_models

from mpi4py import MPI

nside = 4096

map_dist = pysm.MapDistribution(
    pixel_indices=None, nside=nside, mpi_comm=MPI.COMM_WORLD
)

components = []
for comp in ["SO_d0", "SO_s0", "SO_f0", "SO_a0"]:
    components.append(get_so_models(comp, nside, map_dist=map_dist))

components.append(
    so_pysm_models.WebSkyCIB(
        websky_version="0.3",
        interpolation_kind="linear",
        nside=nside,
        map_dist=map_dist,
    )
)

components.append(
    so_pysm_models.WebSkySZ(
        version="0.3",
        nside=nside,
        map_dist=map_dist,
        sz_type="kinetic",
    )
)

components.append(
    so_pysm_models.WebSkySZ(
        version="0.3",
        nside=nside,
        map_dist=map_dist,
        sz_type="thermal",
    )
)

components.append(so_pysm_models.WebSkyCMB(websky_version="0.3", lensed=True, seed=1, nside=nside,map_dist=map_dist))

sky = pysm.Sky(nside=nside, component_objects=components, map_dist=map_dist)

m = sky.get_emission(
    freq=np.arange(50, 55) * u.GHz, weights=np.array([0.1, 0.3, 0.5, 0.3, 0.1])
)

print(map_dist.mpi_comm.rank, m.shape, m.min(), m.max())

m_smoothed = pysm.apply_smoothing_and_coord_transform(
    m, fwhm=1 * u.deg, map_dist=map_dist
)

print(map_dist.mpi_comm.rank, m_smoothed.shape, m_smoothed.min(), m_smoothed.max())

hp.write_map("output_map.fits", m.value)
hp.write_map("output_map_smoothed.fits", m_smoothed.value)
