import numpy as np
import so_pysm_models
from memreport import MemReporter
from mpi4py import MPI
from so_pysm_models import get_so_models

import pysm3 as pysm
import pysm3.units as u

nside = 4096



map_dist = pysm.MapDistribution(
    pixel_indices=None, nside=nside, mpi_comm=MPI.COMM_WORLD
)

import warnings

if map_dist.mpi_comm.rank > 0:
    warnings.filterwarnings("ignore")


memreport = MemReporter(map_dist.mpi_comm)
memreport.run("After imports")

timelines_size_GB = 10
GB_to_bytes = 1024**3
num_elements = timelines_size_GB * GB_to_bytes//np.dtype(np.double).itemsize

timelines = np.ones(num_elements, dtype=np.double)

memreport.run(f"Created fake timelines of {timelines.nbytes/GB_to_bytes:.2f} GB per process")

components = []
for comp in ["SO_d0s", "SO_s0s", "SO_f0s", "SO_a0s"]:
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

components.append(so_pysm_models.WebSkyCMBMap(websky_version="0.3", lensed=True, seed=1, nside=nside,map_dist=map_dist))

memreport.run("Created components")

sky = pysm.Sky(nside=nside, component_objects=components, map_dist=map_dist)

memreport.run("Created Sky")


for each in range(10):
    m = sky.get_emission(
        freq=np.linspace(31, 47, 10) * u.GHz, weights=np.ones(10)
    )

    memreport.run(f"Ran bandpass integration channel {each}")

    #print(map_dist.mpi_comm.rank, m.shape, m.min(), m.max())

    m_smoothed = pysm.apply_smoothing_and_coord_transform(
        m, fwhm=1 * u.deg, map_dist=map_dist
    )

    memreport.run(f"Ran smoothing channel {each}")

    #print(map_dist.mpi_comm.rank, m_smoothed.shape, m_smoothed.min(), m_smoothed.max())

#hp.write_map("output_map.fits", m.value, overwrite=True)
#hp.write_map("output_map_smoothed.fits", m_smoothed.value, overwrite=True)

del sky, m, m_smoothed
memreport.run("Completed")
