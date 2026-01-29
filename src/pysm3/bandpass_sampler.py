"""Bandpass sampling utilities for PySM.

This module provides functions for resampling bandpasses using kernel density
estimation (KDE). The implementation is based on the approach used in MBS 16
for Simon's Observatory bandpass sampling.

The resampling process involves:
1. Normalizing the bandpass to create a probability distribution function (PDF)
2. Computing the cumulative distribution function (CDF)
3. Bootstrap resampling from the CDF
4. Using Gaussian kernel density estimation to create smooth resampled bandpasses
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

try:
    from sklearn.model_selection import GridSearchCV, LeaveOneOut
    from sklearn.neighbors import KernelDensity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def bandpass_distribution_function(bnu, nu):
    """Create an interpolated inverse CDF for bandpass resampling.

    Parameters
    ----------
    bnu : array_like
        Bandpass weights/transmission values
    nu : array_like
        Frequencies in GHz

    Returns
    -------
    callable
        Interpolated inverse cumulative distribution function that maps
        uniform random values [0, 1] to frequencies
    """
    bnu = np.asarray(bnu)
    nu = np.asarray(nu)

    # Normalize to make it a PDF
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    A = trapezoid(bnu, nu)
    if A != 1:
        bnu = bnu / A

    # Interpolate the bandpass
    b = scipy.interpolate.interp1d(x=nu, y=bnu)

    # Estimate the CDF by integrating from min to each frequency
    # Include the first frequency explicitly with CDF value 0
    Pnu = np.concatenate(
        [
            [0.0],
            np.array(
                [scipy.integrate.quad(b, a=nu.min(), b=inu)[0] for inu in nu[1:]]
            ),
        ]
    )

    # Interpolate the inverse CDF over the full frequency range
    Binterp = scipy.interpolate.interp1d(
        Pnu, nu, bounds_error=False, fill_value="extrapolate"
    )
    return Binterp


def search_optimal_kernel_bandwidth(x):
    """Find optimal Gaussian kernel bandwidth using cross-validation.

    Uses leave-one-out cross-validation with a grid search to find the
    optimal bandwidth for kernel density estimation.

    Parameters
    ----------
    x : array_like
        Sample points for which to estimate the optimal bandwidth

    Returns
    -------
    float
        Optimal bandwidth parameter

    Raises
    ------
    ImportError
        If scikit-learn is not installed
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for kernel bandwidth optimization. "
            "Install it with: pip install scikit-learn"
        )

    bandwidths = np.logspace(np.log10(0.1), np.log10(2), 64)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=LeaveOneOut()
    )
    grid.fit(x[:, None])
    return grid.best_params_["bandwidth"]


def bandpass_kresampling(h, nu_i, freq_range, nresample=54):
    """Resample a bandpass using Gaussian kernel density estimation.

    Parameters
    ----------
    h : float
        Kernel bandwidth parameter
    nu_i : array_like
        Bootstrap resampled frequencies in GHz
    freq_range : tuple of float
        (min_freq, max_freq) range for output frequencies in GHz
    nresample : int, optional
        Number of frequency points in the resampled bandpass (default: 54)

    Returns
    -------
    nud : ndarray
        Resampled frequency array in GHz
    resampled_bpass : ndarray
        Resampled bandpass weights (normalized)

    Raises
    ------
    ImportError
        If scikit-learn is not installed
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for kernel density estimation. "
            "Install it with: pip install scikit-learn"
        )

    # Instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=h, kernel="gaussian")
    kde.fit(nu_i[:, None])

    # Create frequency grid
    nud = np.linspace(*freq_range, nresample)

    # score_samples returns the log of the probability density
    resampled_bpass = np.exp(kde.score_samples(nud[:, None]))

    return nud, resampled_bpass


def compute_moments(nu, bnu):
    """Compute the first two moments of a bandpass.

    Parameters
    ----------
    nu : array_like
        Frequencies in GHz
    bnu : array_like
        Bandpass weights/transmission (should be normalized)

    Returns
    -------
    centroid : float
        First moment (mean frequency) in GHz
    bandwidth : float
        Square root of second central moment (standard deviation) in GHz
    """
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    centroid = trapezoid(bnu * nu, nu)
    variance = trapezoid(bnu * (nu - centroid) ** 2, nu)
    bandwidth = np.sqrt(variance)

    return centroid, bandwidth


def resample_bandpass(nu, bnu, num_wafers=1, bootstrap_size=128, random_seed=None):
    """Resample a bandpass to create variations for multiple detectors/wafers.

    This is a high-level convenience function that combines the CDF creation,
    bootstrap resampling, kernel bandwidth optimization, and KDE resampling
    into a single operation.

    Parameters
    ----------
    nu : array_like
        Input frequencies in GHz
    bnu : array_like
        Input bandpass weights/transmission
    num_wafers : int, optional
        Number of resampled bandpasses to generate (default: 1)
    bootstrap_size : int, optional
        Number of bootstrap samples to draw (default: 128).
        Larger values give smoother results but take longer.
    random_seed : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    resampled_bandpasses : list of dict
        List of dictionaries, one per wafer, each containing:
        - 'frequency': ndarray of frequencies in GHz
        - 'weights': ndarray of bandpass weights (normalized)
        - 'centroid': float, mean frequency in GHz
        - 'bandwidth': float, standard deviation in GHz

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple Gaussian bandpass
    >>> nu = np.linspace(90, 110, 100)
    >>> bnu = np.exp(-0.5 * ((nu - 100) / 5)**2)
    >>> # Resample for 4 wafers
    >>> results = resample_bandpass(nu, bnu, num_wafers=4, random_seed=42)
    >>> print(f"Wafer 0 centroid: {results[0]['centroid']:.2f} GHz")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    nu = np.asarray(nu)
    bnu = np.asarray(bnu)

    # Normalize bandpass
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

    bnu = bnu / trapezoid(bnu, nu)

    # Create CDF interpolator
    interpolated_cdf = bandpass_distribution_function(bnu=bnu, nu=nu)

    # Find optimal kernel bandwidth (only once, reuse for all wafers)
    X = np.random.uniform(size=bootstrap_size)
    nu_resampled = interpolated_cdf(X)
    h = search_optimal_kernel_bandwidth(nu_resampled)

    resampled_bandpasses = []

    for i in range(num_wafers):
        # Bootstrap resample from CDF
        X = np.random.uniform(size=bootstrap_size)
        nu_resampled = interpolated_cdf(X)

        # Apply KDE to create smooth bandpass
        freq, weights = bandpass_kresampling(
            h, nu_resampled, freq_range=[nu.min(), nu.max()], nresample=bootstrap_size
        )

        # Normalize
        weights = weights / trapezoid(weights, freq)

        # Compute moments
        centroid, bandwidth = compute_moments(freq, weights)

        resampled_bandpasses.append(
            {
                "frequency": freq,
                "weights": weights,
                "centroid": centroid,
                "bandwidth": bandwidth,
            }
        )

    return resampled_bandpasses
