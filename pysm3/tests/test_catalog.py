from numpy.testing import assert_allclose

from pysm3.models.catalog import evaluate_poly, evaluate_model
import numpy as np


def test_evaluate_poly():
    np.random.seed(100)
    for N in [4, 5, 6]:
        p = np.random.rand(N)
        x = np.random.rand(1)[0]
        assert_allclose(np.polyval(p, x), evaluate_poly(p, x))


def test_evaluate_model_1freq_offset():
    coeff = np.array([[0, 0, 0, 0, 3.7]])
    freqs = np.array([100])
    weights = np.array([0])  # not used when 1 point
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 3.7


def test_evaluate_model_1freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]])
    freqs = np.array([np.exp(3)])
    weights = np.array([0])
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 6
