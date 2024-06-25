from numpy.testing import assert_allclose

from pysm3.models.catalog import evaluate_poly
import numpy as np


def test_evaluate_poly():
    np.random.seed(100)
    for N in [4, 5, 6]:
        p = np.random.rand(N)
        x = np.random.rand(1)[0]
        assert_allclose(np.polyval(p, x), evaluate_poly(p, x))
