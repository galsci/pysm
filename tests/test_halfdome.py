import os

import healpy as hp
import numpy as np
import pytest

try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

from pysm3 import (
    HalfDomeSZ,
    # SPT_CIB_map_scaling,
    # HalfDomeCIB,
    # HalfDomeRadioGalaxies,
    utils,
)  # , WebSkyCMBTensor


@pytest.mark.parametrize("sz_type", ["thermal"]) # , "kinetic"]) (kinetic not implemented yet)
def test_halfdome_sz_seeds(sz_type):

    nside = 128

    tsz_0 = HalfDomeSZ(nside, "0.1", sz_type=sz_type, seed=0)
    tsz_1 = HalfDomeSZ(nside, "0.1", sz_type=sz_type, seed=1)

    freq = 100 * u.GHz
    tsz_0_map = tsz_0.get_emission(freq).to(
        u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)
    )
    tsz_1_map = tsz_1.get_emission(freq).to(
        u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)
    )

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(tsz_0_map, tsz_1_map)

        
@pytest.mark.parametrize("sz_type", ["thermal"]) # , "kinetic"]) (kinetic not implemented yet)
def test_halfdome_sz(tmp_path, monkeypatch, sz_type): # It needs to be the last

    os.environ.pop("PYSM_LOCAL_DATA", None)
    monkeypatch.setattr(utils.data, "PREDEFINED_DATA_FOLDERS", [str(tmp_path)])
    nside = 4
    shape = hp.nside2npix(nside)

    path = tmp_path / "halfdome" / "0.1" / "tsz"
    path.mkdir(parents=True)
    filename = "y_b16_halo_res1_s100.fits" if sz_type == "thermal" else "ksz.fits"
    test_map = np.ones(shape, dtype=np.float32)
    if sz_type == "thermal":
        test_map *= 1e-6
    hp.write_map(path / filename, test_map)

    tsz = HalfDomeSZ(nside, "0.1", sz_type=sz_type, seed=0)

    freq = 100 * u.GHz
    tsz_map = tsz.get_emission(freq).to(
        u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)
    )
    value = -4.109055 * u.uK_CMB if sz_type == "thermal" else 1.0 * u.uK_CMB
    np.testing.assert_allclose(np.ones(len(tsz_map[0])) * value, tsz_map[0], rtol=1e-4)
    np.testing.assert_allclose(np.zeros((2, len(tsz_map[0]))) * u.uK_CMB, tsz_map[1:])
