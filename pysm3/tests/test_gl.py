from pysm3.utils.gauss_legendre import *


def random_alm(lmax, mmax, spin):
    rng = np.random.default_rng(48)
    spin = list(spin)
    ncomp = len(spin)
    res = rng.uniform(-1.0, 1.0, (ncomp, nalm(lmax, mmax))) + 1j * rng.uniform(
        -1.0, 1.0, (ncomp, nalm(lmax, mmax))
    )
    # make a_lm with m==0 real-valued
    res[:, 0 : lmax + 1].imag = 0.0
    # zero a few other values dependent on spin
    for i in range(ncomp):
        ofs = 0
        for s in range(spin[i]):
            res[i, ofs : ofs + spin[i] - s] = 0.0
            ofs += lmax + 1 - s
    return res


def test_map2alm2map():

    import ducc0
    import numpy as np
    from time import time

    # set maximum multipole moment
    lmax = 2047
    # maximum m.
    mmax = lmax
    nthreads = multiprocessing.cpu_count()

    alm = random_alm(lmax, mmax, [0, 2, 2])

    GL_map = gl_alm2map(alm, lmax)
    alm2 = gl_map2alm(GL_map, lmax)

    assert ducc0.misc.l2error(alm, alm2) < 1e-8


def test_map2alm2map_Ionly():

    import ducc0
    import numpy as np
    from time import time

    # set maximum multipole moment
    lmax = 2047
    # maximum m.
    mmax = lmax
    nthreads = multiprocessing.cpu_count()

    alm = random_alm(lmax, mmax, [0])
    assert alm.shape[0] == 1

    GL_map = gl_alm2map(alm, lmax)
    alm2 = gl_map2alm(GL_map, lmax)

    assert ducc0.misc.l2error(alm, alm2) < 1e-8
