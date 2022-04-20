import numpy as np
import multiprocessing

nthreads = multiprocessing.cpu_count()

try:
    import ducc0
except ImportError:
    log.warning(
        "Cannot import ducc0, Gauss Legendre pixelization functionality will not work"
    )
    ducc0 = None


def lmax2nlon(lmax):
    return 2 * lmax + 2


def lmax2nlat(lmax):
    return lmax + 1


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def gl_alm2map(alm, lmax, nlon=None, nlat=None):
    if nlon is None:
        nlon = lmax2nlon(lmax)
    if nlat is None:
        nlat = lmax2nlat(lmax)
    mmax = lmax
    if alm.ndim == 1:
        alm = alm.reshape((1, -1))
    GL_map = np.empty((alm.shape[0], nlat, nlon), dtype=np.double)
    ducc0.sht.experimental.synthesis_2d(
        alm=alm[0:1],
        ntheta=nlat,
        nphi=nlon,
        lmax=lmax,
        mmax=mmax,
        spin=0,
        geometry="GL",
        nthreads=nthreads,
        map=GL_map[0:1],
    )
    if alm.shape[0] > 1:
        ducc0.sht.experimental.synthesis_2d(
            alm=alm[1:3],
            ntheta=nlat,
            nphi=nlon,
            lmax=lmax,
            mmax=mmax,
            spin=2,
            geometry="GL",
            nthreads=nthreads,
            map=GL_map[1:3],
        )
    return GL_map


def gl_map2alm(GL_map, lmax):
    mmax = lmax
    alm = np.empty((GL_map.shape[0], nalm(lmax, mmax)), dtype=np.complex128)

    ducc0.sht.experimental.analysis_2d(
        map=GL_map[0:1],
        lmax=lmax,
        mmax=mmax,
        spin=0,
        geometry="GL",
        nthreads=nthreads,
        alm=alm[0:1],
    )
    if GL_map.shape[0] > 1:
        ducc0.sht.experimental.analysis_2d(
            map=GL_map[1:3],
            lmax=lmax,
            mmax=mmax,
            spin=2,
            geometry="GL",
            nthreads=nthreads,
            alm=alm[1:3],
        )
    return alm
