import healpy as hp
import numpy as np
import sys
from pathlib import Path

alm_filename = sys.argv[1]

alm = hp.read_alm(alm_filename, (1, 2, 3))

datadir = Path("data")

for nside in reversed([1024, 2048, 4096]):
    alm_nside = [hp.almxfl(each, np.ones(3*nside-1)) for each in alm]
    m = hp.alm2map(alm_nside, nside=nside)

    hp.write_map(
        datadir / f"dust_gnilc_hybrid_out_nside{nside}.fits",
        m,
        dtype=np.float32,
        overwrite=True,
    )
