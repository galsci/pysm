"""Tests for bandpass sampler module."""

import numpy as np
import pytest

import pysm3


@pytest.fixture
def gaussian_bandpass():
    """Create a simple Gaussian bandpass for testing."""
    nu = np.linspace(90, 110, 100)
    centroid = 100.0
    sigma = 5.0
    bnu = np.exp(-0.5 * ((nu - centroid) / sigma) ** 2)
    return nu, bnu


def test_compute_moments(gaussian_bandpass):
    """Test moment computation for a Gaussian bandpass."""
    nu, bnu = gaussian_bandpass

    # Normalize
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    bnu_norm = bnu / trapezoid(bnu, nu)

    centroid, bandwidth = pysm3.compute_moments(nu, bnu_norm)

    # For a Gaussian, centroid should be close to 100 and bandwidth close to 5
    assert abs(centroid - 100.0) < 0.1, f"Expected centroid ~100, got {centroid}"
    assert abs(bandwidth - 5.0) < 0.7, f"Expected bandwidth ~5, got {bandwidth}"


def test_bandpass_distribution_function(gaussian_bandpass):
    """Test CDF interpolation."""
    nu, bnu = gaussian_bandpass

    # Normalize
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    bnu = bnu / trapezoid(bnu, nu)

    # Create CDF interpolator
    cdf_interp = pysm3.bandpass_distribution_function(bnu, nu)

    # Sample from uniform distribution
    uniform_samples = np.array([0.1, 0.5, 0.9])
    freq_samples = cdf_interp(uniform_samples)

    # Frequencies should be within the input range
    assert np.all(freq_samples >= nu.min())
    assert np.all(freq_samples <= nu.max())

    # Median sample (0.5) should be close to centroid for symmetric distribution
    assert abs(freq_samples[1] - 100.0) < 1.0


@pytest.mark.skipif(
    not hasattr(pysm3, "search_optimal_kernel_bandwidth"),
    reason="scikit-learn not available",
)
def test_search_optimal_kernel_bandwidth():
    """Test kernel bandwidth optimization."""
    # Create sample data
    np.random.seed(42)
    samples = np.random.normal(100, 5, 50)

    try:
        bandwidth = pysm3.search_optimal_kernel_bandwidth(samples)
        assert bandwidth > 0, "Bandwidth should be positive"
        assert bandwidth < 10, "Bandwidth should be reasonable"
    except ImportError:
        pytest.skip("scikit-learn not installed")


@pytest.mark.skipif(
    not hasattr(pysm3, "bandpass_kresampling"),
    reason="scikit-learn not available",
)
def test_bandpass_kresampling():
    """Test KDE resampling."""
    np.random.seed(42)
    # Create bootstrap samples around 100 GHz
    samples = np.random.normal(100, 5, 100)

    try:
        freq, weights = pysm3.bandpass_kresampling(
            h=1.0, nu_i=samples, freq_range=(90, 110), nresample=50
        )

        assert len(freq) == 50, "Should return requested number of samples"
        assert len(weights) == 50, "Weights should match frequency array"
        assert np.all(weights >= 0), "Weights should be non-negative"
        assert freq[0] == 90, "Frequency range should start at specified minimum"
        assert freq[-1] == 110, "Frequency range should end at specified maximum"
    except ImportError:
        pytest.skip("scikit-learn not installed")


@pytest.mark.skipif(
    not hasattr(pysm3, "resample_bandpass"),
    reason="scikit-learn not available",
)
def test_resample_bandpass(gaussian_bandpass):
    """Test full bandpass resampling workflow."""
    nu, bnu = gaussian_bandpass

    try:
        results = pysm3.resample_bandpass(
            nu, bnu, num_wafers=3, bootstrap_size=64, random_seed=42
        )

        assert len(results) == 3, "Should return requested number of wafers"

        for i, result in enumerate(results):
            assert "frequency" in result
            assert "weights" in result
            assert "centroid" in result
            assert "bandwidth" in result

            # Check that bandpass is normalized
            try:
                from numpy import trapezoid
            except ImportError:
                from numpy import trapz as trapezoid

            integral = trapezoid(result["weights"], result["frequency"])
            assert (
                abs(integral - 1.0) < 0.01
            ), f"Wafer {i} bandpass should be normalized"

            # Centroid should be close to original (100 GHz)
            assert (
                abs(result["centroid"] - 100.0) < 2.0
            ), f"Wafer {i} centroid far from expected"

            # Bandwidth should be reasonable
            assert result["bandwidth"] > 0, f"Wafer {i} bandwidth should be positive"
    except ImportError:
        pytest.skip("scikit-learn not installed")


@pytest.mark.skipif(
    not hasattr(pysm3, "resample_bandpass"),
    reason="scikit-learn not available",
)
def test_resample_bandpass_reproducibility(gaussian_bandpass):
    """Test that resampling with same seed gives same results."""
    nu, bnu = gaussian_bandpass

    try:
        results1 = pysm3.resample_bandpass(
            nu, bnu, num_wafers=2, bootstrap_size=64, random_seed=12345
        )
        results2 = pysm3.resample_bandpass(
            nu, bnu, num_wafers=2, bootstrap_size=64, random_seed=12345
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results1[0]["frequency"], results2[0]["frequency"]
        )
        np.testing.assert_array_almost_equal(
            results1[0]["weights"], results2[0]["weights"]
        )
        assert results1[0]["centroid"] == results2[0]["centroid"]
        assert results1[0]["bandwidth"] == results2[0]["bandwidth"]
    except ImportError:
        pytest.skip("scikit-learn not installed")


@pytest.mark.skipif(
    not hasattr(pysm3, "resample_bandpass"),
    reason="scikit-learn not available",
)
def test_resample_bandpass_different_seeds(gaussian_bandpass):
    """Test that different seeds give different results."""
    nu, bnu = gaussian_bandpass

    try:
        results1 = pysm3.resample_bandpass(
            nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=111
        )
        results2 = pysm3.resample_bandpass(
            nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=222
        )

        # Results should be different
        assert not np.allclose(results1[0]["weights"], results2[0]["weights"])
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_bandpass_normalization():
    """Test that unnormalized input gets normalized."""
    nu = np.linspace(90, 110, 50)
    # Create unnormalized bandpass
    bnu = np.ones(50) * 10.0  # Integral is definitely not 1

    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    # Should not raise an error and should normalize internally
    try:
        results = pysm3.resample_bandpass(
            nu, bnu, num_wafers=1, bootstrap_size=64, random_seed=42
        )
        # Output should still be normalized
        integral = trapezoid(results[0]["weights"], results[0]["frequency"])
        assert abs(integral - 1.0) < 0.01
    except ImportError:
        pytest.skip("scikit-learn not installed")
