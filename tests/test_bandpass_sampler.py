import numpy as np
import pytest

import pysm3


@pytest.fixture
def gaussian_bandpass():
    nu = np.linspace(90.0, 110.0, 201)
    centroid = 100.0
    sigma = 5.0
    bnu = np.exp(-0.5 * ((nu - centroid) / sigma) ** 2)
    return nu, bnu


def test_compute_moments_normalizes(gaussian_bandpass):
    nu, bnu = gaussian_bandpass
    c, bw = pysm3.compute_moments(nu, bnu * 123.0)
    assert abs(c - 100.0) < 0.1
    assert abs(bw - 5.0) < 0.7


def test_bandpass_distribution_function_inverse_cdf(gaussian_bandpass):
    nu, bnu = gaussian_bandpass
    inv = pysm3.bandpass_distribution_function(bnu, nu)

    xs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    f = inv(xs)

    assert np.all(np.isfinite(f))
    assert f[0] == pytest.approx(nu.min())
    assert f[-1] == pytest.approx(nu.max())
    assert np.all(np.diff(f) >= 0)
    assert abs(f[2] - 100.0) < 1.0


def test_bandpass_distribution_function_handles_flat_cdf_regions():
    nu = np.linspace(80.0, 120.0, 9)
    bnu = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=float)
    inv = pysm3.bandpass_distribution_function(bnu, nu)
    f = inv(np.linspace(0.0, 1.0, 17))
    assert f.min() >= nu.min()
    assert f.max() <= nu.max()
    assert np.all(np.diff(f) >= 0)


def test_search_optimal_kernel_bandwidth_returns_positive():
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, 5.0, size=64)
    h = pysm3.search_optimal_kernel_bandwidth(x)
    assert np.isfinite(h)
    assert h > 0


def test_bandpass_kresampling_grid_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, 5.0, size=100)
    nud, w = pysm3.bandpass_kresampling(1.0, x, (90.0, 110.0), nresample=55)
    assert nud.shape == (55,)
    assert w.shape == (55,)
    assert nud[0] == pytest.approx(90.0)
    assert nud[-1] == pytest.approx(110.0)
    assert np.all(w >= 0)


def test_resample_bandpass_reproducible(gaussian_bandpass):
    nu, bnu = gaussian_bandpass
    r1 = pysm3.resample_bandpass(nu, bnu, num_wafers=2, bootstrap_size=64, random_seed=1)
    r2 = pysm3.resample_bandpass(nu, bnu, num_wafers=2, bootstrap_size=64, random_seed=1)

    for i in range(2):
        np.testing.assert_allclose(r1[i]["frequency"], r2[i]["frequency"])
        np.testing.assert_allclose(r1[i]["weights"], r2[i]["weights"])
        assert r1[i]["centroid"] == pytest.approx(r2[i]["centroid"])
        assert r1[i]["bandwidth"] == pytest.approx(r2[i]["bandwidth"])


def test_resample_bandpass_changes_with_seed(gaussian_bandpass):
    nu, bnu = gaussian_bandpass
    r1 = pysm3.resample_bandpass(nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=1)
    r2 = pysm3.resample_bandpass(nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=2)
    assert not np.allclose(r1[0]["weights"], r2[0]["weights"])


def test_resample_bandpass_output_is_normalized(gaussian_bandpass):
    nu, bnu = gaussian_bandpass
    out = pysm3.resample_bandpass(nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=3)
    freq = out[0]["frequency"]
    w = out[0]["weights"]

    try:
        from numpy import trapezoid
    except ImportError:  # pragma: no cover
        from numpy import trapz as trapezoid

    integ = trapezoid(w, freq)
    assert integ == pytest.approx(1.0, abs=1e-3)


def test_invalid_inputs_raise():
    nu = np.array([1.0, 0.0, 2.0])
    bnu = np.ones_like(nu)
    with pytest.raises(ValueError):
        pysm3.bandpass_distribution_function(bnu, nu)

    nu2 = np.array([1.0, 2.0, 3.0])
    bnu2 = np.array([1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        pysm3.bandpass_distribution_function(bnu2, nu2)

    with pytest.raises(ValueError):
        pysm3.bandpass_kresampling(-1.0, np.array([1.0, 2.0]), (0.0, 1.0))

