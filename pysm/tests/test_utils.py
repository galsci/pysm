import numpy as np
import pysm
import pysm.units as u


def test_has_polarization():
    h = pysm.utils.has_polarization

    m = np.empty(12)
    assert h(np.empty((3, 12)))
    assert not h(np.empty((1, 12)))
    assert not h(m)
    assert h(np.empty((4, 3, 12)))
    assert not h(np.empty((4, 1, 12)))
    assert h((m, m, m))
    assert h([(m, m, m), (m, m, m)])


def test_bandpass_unit_conversion():
    freqs = np.array([250, 300, 350]) * u.GHz
    weights = np.ones(len(freqs))
    norm_weights = pysm.normalize_weights(freqs.value, weights)
    conversion_factor = pysm.utils.bandpass_unit_conversion(freqs, weights, "uK_CMB")

    each_factor = [
        (1 * u.uK_RJ).to_value(u.uK_CMB, equivalencies=u.cmb_equivalencies(f))
        for f in freqs
    ]
    expected_factor = np.trapz(each_factor * norm_weights, freqs.value)

    np.testing.assert_allclose(expected_factor, conversion_factor.value)
