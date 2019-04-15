import numpy as np
import pysm

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
