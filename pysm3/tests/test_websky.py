import numpy as np
from scipy import stats
import healpy as hp
import pytest

try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

from .. import utils
from .. import WebSkyCIB, WebSkySZ  # , WebSkyCMBTensor


def test_cib(tmp_path):

    nside = 4
    shape = hp.nside2npix(nside)

    path = tmp_path / "websky" / "0.4" / "cib"
    path.mkdir(parents=True)
    hp.write_map(path / "cib_0090.2.fits", np.zeros(shape, dtype=np.float32))
    hp.write_map(path / "cib_0100.0.fits", np.ones(shape, dtype=np.float32))

    interp = WebSkyCIB(
        "0.4",
        "uK_RJ",
        nside,
        interpolation_kind="linear",
        local_folder=tmp_path,
    )

    interpolated_map = interp.get_emission(97 * u.GHz)
    np.testing.assert_allclose(
        np.interp(97, [90.2, 100], [0, 1]) * np.ones(shape) * u.uK_RJ,
        interpolated_map[0],
    )
    np.testing.assert_allclose(
        0 * u.uK_RJ,
        interpolated_map[1:],
    )


@pytest.mark.parametrize("sz_type", ["thermal", "kinetic"])
def test_sz(tmp_path, monkeypatch, sz_type):

    monkeypatch.setattr(utils.data, "PREDEFINED_DATA_FOLDERS", [str(tmp_path)])
    nside = 4
    shape = hp.nside2npix(nside)

    path = tmp_path / "websky" / "0.4"
    path.mkdir(parents=True)
    filename = "tsz_8192_hp.fits" if sz_type == "thermal" else "ksz.fits"
    test_map = np.ones(shape, dtype=np.float32)
    if sz_type == "thermal":
        test_map *= 1e-6
    hp.write_map(path / filename, test_map)

    tsz = WebSkySZ("0.4", sz_type=sz_type, nside=nside)

    tsz_map = tsz.get_emission(100 * u.GHz)
    value = -3.193671 * u.uK_RJ if sz_type == "thermal" else 0.7772276 * u.uK_RJ
    np.testing.assert_allclose(np.ones(len(tsz_map[0])) * value, tsz_map[0], rtol=1e-4)
    np.testing.assert_allclose(np.zeros((2, len(tsz_map[0]))) * u.uK_RJ, tsz_map[1:])


# @pytest.mark.parametrize("tensor_to_scalar", [1, 1e-3])
# def test_cmb_tensor(tmp_path, monkeypatch, tensor_to_scalar):
#
#    monkeypatch.setattr(utils, "PREDEFINED_DATA_FOLDERS", {"C": [str(tmp_path)]})
#    nside = 256
#    lmax = 512
#
#    path = tmp_path / "websky" / "0.3"
#    path.mkdir(parents=True)
#
#    input_cl = np.zeros((6, lmax + 1), dtype=np.double)
#    input_cl[1] = 1e5 * stats.norm.pdf(np.arange(lmax + 1), 250, 30)  # EE
#    filename = path / "tensor_cl_r1_nt0.fits"
#
#    hp.write_cl(filename, input_cl, overwrite=True)
#
#    cmb_tensor = WebSkyCMBTensor("0.3", nside=nside, tensor_to_scalar=tensor_to_scalar)
#
#    freq = 100 * u.GHz
#    cmb_tensor_map = cmb_tensor.get_emission(freq)
#    cmb_tensor_map = cmb_tensor_map.to(
#        u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)
#    )
#
#    cl = hp.anafast(cmb_tensor_map, use_pixel_weights=True, lmax=lmax)
#    # anafast returns results in new ordering
#    # TT, EE, BB, TE, EB, TB
#    np.testing.assert_allclose(
#        input_cl[5][200:300] * tensor_to_scalar, cl[2][200:300], rtol=0.2
#    )
#    np.testing.assert_allclose(0, cl[:2], rtol=1e-3)
#    np.testing.assert_allclose(0, cl[3:], rtol=1e-3, atol=1e-4)
