from urllib.error import URLError

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
import pixell.enmap
import pytest
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose

import pysm3
import pysm3.units as u
from pysm3 import utils


def test_get_relevant_frequencies():
    freqs = [10, 11, 14, 16, 20]
    assert utils.get_relevant_frequencies(freqs, 11, 14) == [11, 14]
    assert utils.get_relevant_frequencies(freqs, 11.5, 14) == [11, 14]
    assert utils.get_relevant_frequencies(freqs, 11.5, 13.9) == [11, 14]
    assert utils.get_relevant_frequencies(freqs, 11, 14.1) == [11, 14, 16]
    assert utils.get_relevant_frequencies(freqs, 10, 10.1) == [10, 11]
    assert utils.get_relevant_frequencies(freqs, 10, 19) == freqs
    assert utils.get_relevant_frequencies(freqs, 15, 19) == [14, 16, 20]


def test_has_polarization():
    h = pysm3.utils.has_polarization

    m = np.empty(12)
    assert h(np.empty((3, 12)))
    assert not h(np.empty((1, 12)))
    assert not h(m)
    assert h(np.empty((4, 3, 12)))
    assert not h(np.empty((4, 1, 12)))
    assert h((m, m, m))
    assert h([(m, m, m), (m, m, m)])


def test_bandpass_unit_conversion():
    nside = 32
    freqs = np.array([250, 300, 350]) * u.GHz
    weights = np.ones(len(freqs))
    sky = pysm3.Sky(nside=nside, preset_strings=["c2"])
    CMB_rj_int = sky.get_emission(freqs, weights)
    CMB_thermo_int = CMB_rj_int * pysm3.utils.bandpass_unit_conversion(
        freqs, weights, u.uK_CMB
    )
    expected_map = pysm3.read_map(
        "pysm_2/lensed_cmb.fits", field=(0, 1), nside=nside, unit=u.uK_CMB
    )
    for pol in [0, 1]:
        assert_quantity_allclose(expected_map[pol], CMB_thermo_int[pol], rtol=1e-4)


def test_bandpass_integration_tophat():
    input_map = np.ones(12, dtype=np.double)
    output_map = np.zeros_like(input_map)
    freqs = [99, 100, 101] * u.GHz
    weights = None
    freqs = utils.check_freq_input(freqs)
    weights = utils.normalize_weights(freqs, weights)
    for i, (_freq, _weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(freqs, weights, i, input_map, output_map)
    np.testing.assert_allclose(input_map, output_map)


@pytest.mark.parametrize("freq_spacing", ["uniform", "non-uniform"])
def test_trapz(freq_spacing):
    freqs = [99, 100, 101] * u.GHz
    if freq_spacing == "non-uniform":
        freqs[-1] += 30 * u.GHz
    input_maps = np.array([1, 1.5, 1.2], dtype=np.double)
    output_map = np.array([0], dtype=np.double)
    weights = [0.3, 1, 0.3]
    freqs = utils.check_freq_input(freqs)
    weights = utils.normalize_weights(freqs, weights)
    for i, (_freq, _weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(freqs, weights, i, input_maps[i : i + 1], output_map)

    expected = trapezoid(weights * input_maps, freqs)
    np.testing.assert_allclose(expected, output_map)


def test_bandpass_integration_weights():
    input_map = np.ones(12, dtype=np.double)
    output_map = np.zeros_like(input_map)
    freqs = [99, 100, 101] * u.GHz
    weights = [0.3, 1, 0.3]
    freqs = utils.check_freq_input(freqs)
    weights = utils.normalize_weights(freqs, weights)
    for i, (_freq, _weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(freqs, weights, i, input_map, output_map)
    np.testing.assert_allclose(input_map, output_map)


def test_remotedata(tmp_path, monkeypatch):
    data_folder = tmp_path / "data"
    data_folder.mkdir()
    test_file = data_folder / "testfile.txt"
    test_file.touch()
    monkeypatch.setenv("PYSM_LOCAL_DATA", str(data_folder))
    filename = pysm3.utils.RemoteData().get("testfile.txt")
    assert filename == str(test_file)


def test_remotedata_globalpath(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()
    test_file = data_folder / "testfile.txt"
    test_file.touch()
    filename = pysm3.utils.RemoteData().get(str(test_file))
    assert filename == str(test_file)


@pytest.fixture
def test_fits_file(tmp_path):
    d = tmp_path / "sub"
    c1 = fits.Column(name="a", array=np.array([1, 2]), format="K")
    c2 = fits.Column(name="b", array=np.array([4, 5]), format="K")
    c3 = fits.Column(name="c", array=np.array([7, 8]), format="K")
    t = fits.BinTableHDU.from_columns([c1, c2, c3])
    t.writeto(d)
    return d


def test_add_metadata(test_fits_file):
    pysm3.utils.add_metadata(
        [test_fits_file, test_fits_file],
        field=1,
        coord="G",
        unit="uK_RJ",
        ref_freq="353 GHz",
    )
    with fits.open(test_fits_file) as f:
        assert f[1].header["COORDSYS"] == "G"
        assert f[1].header["TUNIT1"] == "uK_RJ"
        assert f[1].header["TUNIT2"] == "uK_RJ"
        assert f[1].header["TUNIT3"] == "uK_RJ"
        assert f[1].header["REF_FREQ"] == "353 GHz"


def test_add_metadata_different_units(test_fits_file):
    pysm3.utils.add_metadata(
        [test_fits_file],
        field=1,
        coord="G",
        unit=["uK_RJ", "mK_RJ", "K_CMB"],
        ref_freq="353 GHz",
    )
    with fits.open(test_fits_file) as f:
        assert f[1].header["COORDSYS"] == "G"
        assert f[1].header["TUNIT1"] == "uK_RJ"
        assert f[1].header["TUNIT2"] == "mK_RJ"
        assert f[1].header["TUNIT3"] == "K_CMB"
        assert f[1].header["REF_FREQ"] == "353 GHz"


def test_data_raise():
    with pytest.raises(URLError):
        pysm3.utils.RemoteData().get("doesntexist.txt")


class ReturnsCar:
    def __init__(self, wcs):
        self.wcs = wcs

    def get_emission(self):
        emission = np.ones(12) * u.uK_RJ
        return utils.wrap_wcs(emission, self.wcs)


def test_wrap_wcs_no_wcs():
    EmissionModel = ReturnsCar(wcs=None)
    emission = EmissionModel.get_emission()
    assert emission.unit == u.uK_RJ
    assert not hasattr(emission, "wcs")


def test_wrap_wcs_with_wcs():
    shape, wcs = pixell.enmap.fullsky_geometry(
        (10 * u.deg).to_value(u.rad),
        dims=(3,),
        variant="fejer1",
    )
    EmissionModel = ReturnsCar(wcs=wcs)
    emission = EmissionModel.get_emission()
    assert hasattr(emission, "wcs")
