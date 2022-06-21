from pysm3.utils.gauss_legendre import *
import healpy as hp
import numpy as np


def test_clip_pad():

    lmax = 64
    alm_size = hp.Alm.getsize(lmax)
    alm = np.arange(1, alm_size + 1, dtype=np.complex128)
    alm = alm.reshape((1, -1))

    higher_lmax = 128
    alm_padded = pad_alm(alm, higher_lmax)

    assert alm_padded.shape[-1] == hp.Alm.getsize(higher_lmax)
    assert (alm_padded != 0).sum() == alm_size

    clip_indices = clip_alm(higher_lmax, lmax)
    alm_clipped = alm_padded[:, clip_indices]
    np.testing.assert_array_equal(alm, alm_clipped)


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

    # set maximum multipole moment
    lmax = 2047
    # maximum m.
    mmax = lmax

    alm = random_alm(lmax, mmax, [0, 2, 2])

    GL_map = gl_alm2map(alm, lmax)
    alm2 = gl_map2alm(GL_map, lmax)

    assert ducc0.misc.l2error(alm, alm2) < 1e-8


def test_map2alm2map_Ionly():

    import ducc0

    # set maximum multipole moment
    lmax = 2047
    # maximum m.
    mmax = lmax

    alm = random_alm(lmax, mmax, [0])
    assert alm.shape[0] == 1

    GL_map = gl_alm2map(alm, lmax)
    alm2 = gl_map2alm(GL_map, lmax)

    assert ducc0.misc.l2error(alm, alm2) < 1e-8
