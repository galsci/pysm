import libsharp
import numpy as np
from mpi4py import MPI

lmax = 10
mpi_comm = MPI.COMM_WORLD
local_m_indices = np.arange(mpi_comm.rank, lmax + 1, mpi_comm.size, dtype=np.int32)
order = libsharp.packed_real_order(lmax, ms=local_m_indices)
print(MPI.COMM_WORLD.rank, order.mval(), order.mvstart(), order.local_size())

