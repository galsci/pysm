import pysm3 as pysm
from numpy.testing import assert_allclose

from pysm3.models.catalog import evaluate_poly, evaluate_model
import numpy as np


def test_evaluate_poly():
    np.random.seed(100)
    for N in [4, 5, 6]:
        p = np.random.rand(N)
        x = np.random.rand(1)[0]
        assert_allclose(np.polyval(p, x), evaluate_poly(p, x))


def test_evaluate_model_1freq_flat():
    coeff = np.array([[0, 0, 0, 0, 3.7]])
    freqs = np.array([100])
    weights = np.array([0])  # not used when 1 point
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 3.7


def test_evaluate_model_1freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]])
    freqs = np.array([np.exp(3)])
    weights = np.array([0])
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 6


def test_evaluate_model_2freq_flat():
    coeff = np.array([[0, 0, 0, 0, 3.7]])
    freqs = np.exp(np.array([3, 4]))  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1])
    weights = pysm.normalize_weights(freqs, weights)
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 3.7


def test_evaluate_model_2freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]])
    freqs = np.exp(np.array([3, 4]))  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1])
    weights = pysm.normalize_weights(freqs, weights)
    flux = evaluate_model(freqs, weights, coeff)[0]
    assert flux > 6
    assert flux < 8
    assert flux == np.trapz(weights * np.array([6, 8]), x=freqs)
