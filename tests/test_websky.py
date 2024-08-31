import os

import healpy as hp
import numpy as np
import pytest

try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

from pysm3 import (
    SPT_CIB_map_scaling,
    WebSkyCIB,
    WebSkyRadioGalaxies,
    WebSkySZ,
    utils,
)  # , WebSkyCMBTensor


def test_SPT_CIB_map_scaling():
    frequencies = """18.7000 21.6000 24.5000 27.3000 30.0000 35.9000
    41.7000 44.0000 47.4000 63.9000 67.8000 70.0000 73.7000 79.6000
    90.2000 100.0000 111.0000 129.0000 143.0000 153.0000 164.0000 189.0000
    210.0000 217.0000 232.0000 256.0000 275.0000 294.0000 306.0000 314.0000
    340.0000 353.0000 375.0000 409.0000 467.0000 525.0000 545.0000 584.0000
    643.0000 729.0000 817.0000 857.0000 906.0000 994.0000 1080.0000"""
    freq = np.array(list(map(float, frequencies.split())))
    expected_scaling = """1.3382 1.3375 1.3368 1.3361 1.3354
    1.3338 1.3321 1.3314 1.3303 1.3245
    1.3229 1.3220 1.3205 1.3179 1.3127 1.3075 1.3010 1.2888 1.2780 1.2696
    1.2596 1.2346 1.2114 1.2033 1.1858 1.1575 1.1358 1.1153 1.1033 1.0957
    1.0735 1.0639 1.0500 1.0335 1.0163 1.0077 1.0059 1.0036 1.0016 1.0005
    1.0002 1.0001 1.0000 1.0000 1.0000"""
    expected = np.array(list(map(float, expected_scaling.split())))
    np.testing.assert_allclose(SPT_CIB_map_scaling(freq), expected, rtol=1e-4)


def test_radiogalaxies(tmp_path):

    nside = 4
    shape = hp.nside2npix(nside)

    path = tmp_path / "websky" / "0.4" / "radio"
    path.mkdir(parents=True)
    hp.write_map(path / "radio_0090.2.fits", np.zeros(shape, dtype=np.float32))
    hp.write_map(path / "radio_0100.0.fits", np.ones(shape, dtype=np.float32))

    interp = WebSkyRadioGalaxies(
        nside,
        "0.4",
        "uK_RJ",
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


def test_cib(tmp_path):

    nside = 4
    shape = hp.nside2npix(nside)

    path = tmp_path / "websky" / "0.4" / "cib"
    path.mkdir(parents=True)
    hp.write_map(path / "cib_0090.2.fits", np.zeros(shape, dtype=np.float32))
    hp.write_map(path / "cib_0100.0.fits", np.ones(shape, dtype=np.float32))

    interp = WebSkyCIB(
        nside,
        "0.4",
        "uK_RJ",
        interpolation_kind="linear",
        local_folder=tmp_path,
        apply_SPT_correction=False,
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

    interp = WebSkyCIB(
        nside,
        "0.4",
        "uK_RJ",
        interpolation_kind="linear",
        local_folder=tmp_path,
        apply_SPT_correction=True,
    )

    interpolated_map = interp.get_emission(97 * u.GHz)
    np.testing.assert_allclose(
        1.3075 * np.interp(97, [90.2, 100], [0, 1]) * np.ones(shape) * u.uK_RJ,
        interpolated_map[0],
        rtol=1e-4,
    )


@pytest.mark.parametrize("sz_type", ["thermal", "kinetic"])
def test_sz(tmp_path, monkeypatch, sz_type):

    os.environ.pop("PYSM_LOCAL_DATA", None)
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

    tsz = WebSkySZ(nside, "0.4", sz_type=sz_type)

    freq = 100 * u.GHz
    tsz_map = tsz.get_emission(freq).to(
        u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)
    )
    value = -4.109055 * u.uK_CMB if sz_type == "thermal" else 1.0 * u.uK_CMB
    np.testing.assert_allclose(np.ones(len(tsz_map[0])) * value, tsz_map[0], rtol=1e-4)
    np.testing.assert_allclose(np.zeros((2, len(tsz_map[0]))) * u.uK_CMB, tsz_map[1:])


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
