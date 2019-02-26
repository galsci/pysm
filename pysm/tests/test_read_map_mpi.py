import pytest
import pysm

try:
    from mpi4py import MPI
except ImportError:
    pytest.skip("mpi4py failed to import, skip MPI tests", allow_module_level=True)


@pytest.fixture
def setup_mpi_communicator():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank


def test_read_map_mpi(setup_mpi_communicator):
    comm, rank = setup_mpi_communicator
    # Reads pixel [0] on rank 0
    # pixels [0,1] on rank 1
    # pixels [0,1,2] on rank 2 and so on.
    m = pysm.read_map(
        "pysm_2/dust_temp.fits",
        nside=8,
        field=0,
        pixel_indices=list(range(0, rank + 1)),
        mpi_comm=comm,
    )
    assert len(m) == rank + 1
