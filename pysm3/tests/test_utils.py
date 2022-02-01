import numpy as np
import pysm3
import pysm3.units as u
from astropy.tests.helper import assert_quantity_allclose
from pysm3 import utils


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
    for i, (freq, weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(freqs, weights, i, input_map, output_map)
    np.testing.assert_allclose(input_map, output_map)


def test_bandpass_integration_weights():
    input_map = np.ones(12, dtype=np.double)
    output_map = np.zeros_like(input_map)
    freqs = [99, 100, 101] * u.GHz
    weights = [0.3, 1, 0.3]
    freqs = utils.check_freq_input(freqs)
    weights = utils.normalize_weights(freqs, weights)
    for i, (freq, weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(freqs, weights, i, input_map, output_map)
    np.testing.assert_allclose(input_map, output_map)


def test_remotedata(tmp_path):
    import os

    data_folder = tmp_path / "data"
    data_folder.mkdir()
    test_file = data_folder / "testfile.txt"
    test_file.touch()
    os.environ["PYSM_LOCAL_DATA"] = str(data_folder)
    filename = pysm3.utils.RemoteData().get("testfile.txt")
    assert filename == str(test_file)
