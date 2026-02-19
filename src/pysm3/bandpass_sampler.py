"""Bandpass sampling utilities.

This module implements a lightweight bandpass resampling workflow inspired by
SO/MBS-style "bandpass sampling":

1. Interpret an input bandpass as a PDF over frequency (after normalization).
2. Draw bootstrap samples from the inverse CDF.
3. Build a smooth resampled bandpass via Gaussian KDE on a fixed grid.

Implementation notes
--------------------
PySM should not require scikit-learn at runtime. Therefore KDE evaluation and
bandwidth selection are implemented using NumPy/SciPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np

try:
    from numpy import trapezoid
except ImportError:  # pragma: no cover (NumPy < 1.20)
    from numpy import trapz as trapezoid

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _as_1d_float_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} must have at least 2 elements, got {arr.size}")
    return arr


def _validate_bandpass(nu_ghz: np.ndarray, bnu: np.ndarray) -> None:
    if nu_ghz.shape != bnu.shape:
        raise ValueError(
            f"nu and bnu must have the same shape, got {nu_ghz.shape} and {bnu.shape}"
        )
    if not np.all(np.isfinite(nu_ghz)) or not np.all(np.isfinite(bnu)):
        raise ValueError("nu and bnu must be finite")
    if np.any(np.diff(nu_ghz) <= 0):
        raise ValueError("nu must be strictly increasing")
    if np.any(bnu < 0):
        raise ValueError("bnu must be non-negative")
    area = trapezoid(bnu, nu_ghz)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("bandpass integral must be positive")


def bandpass_distribution_function(bnu, nu) -> Callable[[np.ndarray], np.ndarray]:
    """Create an interpolated inverse CDF for bandpass resampling.

    Parameters
    ----------
    bnu : array_like
        Bandpass weights/transmission values (non-negative).
    nu : array_like
        Frequencies in GHz (strictly increasing).

    Returns
    -------
    callable
        Inverse CDF mapping uniform random values in [0, 1] to frequencies (GHz).
    """
    nu_ghz = _as_1d_float_array(nu, "nu")
    bnu_arr = _as_1d_float_array(bnu, "bnu")
    _validate_bandpass(nu_ghz, bnu_arr)

    pdf = bnu_arr / trapezoid(bnu_arr, nu_ghz)

    cdf = cumulative_trapezoid(pdf, nu_ghz, initial=0.0)
    if cdf[-1] <= 0:
        raise ValueError("invalid CDF (non-positive final value)")
    cdf = cdf / cdf[-1]

    # interp1d requires a strictly increasing x; flat regions happen when pdf=0.
    cdf_unique, idx = np.unique(cdf, return_index=True)
    nu_unique = nu_ghz[idx]

    return interp1d(
        cdf_unique,
        nu_unique,
        bounds_error=False,
        fill_value=(nu_ghz.min(), nu_ghz.max()),
        assume_sorted=True,
    )


def compute_moments(nu, bnu) -> Tuple[float, float]:
    """Compute centroid and bandwidth (stddev) of a bandpass.

    Parameters
    ----------
    nu : array_like
        Frequencies in GHz (strictly increasing).
    bnu : array_like
        Bandpass weights (non-negative). If not normalized, it will be normalized.

    Returns
    -------
    centroid : float
        First moment (mean frequency), in GHz.
    bandwidth : float
        Standard deviation around the centroid, in GHz.
    """
    nu_ghz = _as_1d_float_array(nu, "nu")
    bnu_arr = _as_1d_float_array(bnu, "bnu")
    _validate_bandpass(nu_ghz, bnu_arr)

    pdf = bnu_arr / trapezoid(bnu_arr, nu_ghz)
    centroid = float(trapezoid(pdf * nu_ghz, nu_ghz))
    variance = float(trapezoid(pdf * (nu_ghz - centroid) ** 2, nu_ghz))
    return centroid, float(np.sqrt(max(variance, 0.0)))


def bandpass_kresampling(
    h: float,
    nu_i,
    freq_range: Tuple[float, float],
    nresample: int = 54,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a bandpass with a Gaussian KDE evaluated on a grid.

    Parameters
    ----------
    h : float
        KDE bandwidth (Gaussian sigma), in GHz.
    nu_i : array_like
        Bootstrap resampled frequencies in GHz (1D).
    freq_range : (float, float)
        (min_freq, max_freq) of output grid in GHz.
    nresample : int
        Number of points in the output grid.

    Returns
    -------
    nud : ndarray
        Output frequency grid (GHz).
    resampled_bpass : ndarray
        KDE density evaluated on nud (not normalized over the finite grid).
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite float")
    if nresample < 2:
        raise ValueError("nresample must be >= 2")

    nu_samples = np.asarray(nu_i, dtype=float)
    if nu_samples.ndim != 1 or nu_samples.size < 2:
        raise ValueError("nu_i must be a 1D array with at least 2 samples")
    if not np.all(np.isfinite(nu_samples)):
        raise ValueError("nu_i must be finite")

    fmin, fmax = float(freq_range[0]), float(freq_range[1])
    if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin):
        raise ValueError("freq_range must be finite and satisfy max > min")

    nud = np.linspace(fmin, fmax, int(nresample), dtype=float)
    # Gaussian KDE: mean of kernels at each grid point.
    z = (nud[:, None] - nu_samples[None, :]) / h
    resampled = np.exp(-0.5 * z**2).mean(axis=1) / (h * _SQRT_2PI)
    return nud, resampled


def search_optimal_kernel_bandwidth(
    x,
    bandwidths: Iterable[float] | None = None,
) -> float:
    """Select KDE bandwidth by leave-one-out log-likelihood grid search.

    Parameters
    ----------
    x : array_like
        1D samples (GHz).
    bandwidths : iterable of float, optional
        Candidate bandwidths to evaluate. If None, a reasonable log-spaced
        grid is built from the sample standard deviation.

    Returns
    -------
    float
        Best bandwidth (GHz).
    """
    xs = np.asarray(x, dtype=float)
    if xs.ndim != 1 or xs.size < 3:
        raise ValueError("x must be a 1D array with at least 3 samples")
    if not np.all(np.isfinite(xs)):
        raise ValueError("x must be finite")

    if bandwidths is None:
        s = float(np.std(xs, ddof=1))
        # Fall back to a small bandwidth if samples are identical.
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        lo = max(1e-3, 0.05 * s)
        hi = 2.0 * s
        bandwidths = np.logspace(np.log10(lo), np.log10(hi), 64)
    else:
        bandwidths = np.asarray(list(bandwidths), dtype=float)

    bandwidths = np.asarray(bandwidths, dtype=float)
    if bandwidths.ndim != 1 or bandwidths.size == 0:
        raise ValueError("bandwidths must be a 1D non-empty iterable")
    if np.any(~np.isfinite(bandwidths)) or np.any(bandwidths <= 0):
        raise ValueError("bandwidths must be positive and finite")

    # LOO log-likelihood for Gaussian KDE.
    n = xs.size
    d2 = (xs[:, None] - xs[None, :]) ** 2
    # Exclude self-contribution.
    np.fill_diagonal(d2, np.inf)

    best_h = float(bandwidths[0])
    best_ll = -np.inf
    eps = 1e-300
    for h in bandwidths:
        k = np.exp(-0.5 * d2 / (h * h))
        dens = k.sum(axis=1) / ((n - 1) * h * _SQRT_2PI)
        ll = float(np.log(dens + eps).sum())
        if ll > best_ll:
            best_ll = ll
            best_h = float(h)

    return best_h


@dataclass(frozen=True)
class ResampledBandpass:
    """Container for a resampled bandpass."""

    frequency: np.ndarray
    weights: np.ndarray
    centroid: float
    bandwidth: float


def resample_bandpass(
    nu,
    bnu,
    num_wafers: int = 1,
    bootstrap_size: int = 128,
    random_seed: int | None = None,
) -> list[dict]:
    """Resample an input bandpass to create multiple "wafer" variations.

    Parameters
    ----------
    nu : array_like
        Input frequencies in GHz (strictly increasing).
    bnu : array_like
        Input weights (non-negative).
    num_wafers : int
        Number of resampled bandpasses to generate.
    bootstrap_size : int
        Number of bootstrap samples (also used as the output grid size).
    random_seed : int, optional
        Seed for deterministic output.

    Returns
    -------
    list of dict
        Each dict has keys: 'frequency', 'weights', 'centroid', 'bandwidth'.
    """
    if num_wafers < 1:
        raise ValueError("num_wafers must be >= 1")
    if bootstrap_size < 8:
        raise ValueError("bootstrap_size must be >= 8")

    nu_ghz = _as_1d_float_array(nu, "nu")
    bnu_arr = _as_1d_float_array(bnu, "bnu")
    _validate_bandpass(nu_ghz, bnu_arr)

    rng = np.random.default_rng(random_seed)

    # Build inverse CDF once.
    inv_cdf = bandpass_distribution_function(bnu_arr, nu_ghz)

    # Pick bandwidth from one bootstrap sample (then reuse for all wafers).
    nu_resampled0 = inv_cdf(rng.uniform(0.0, 1.0, size=int(bootstrap_size)))
    h = search_optimal_kernel_bandwidth(nu_resampled0)

    out: list[dict] = []
    for _ in range(int(num_wafers)):
        nu_resampled = inv_cdf(rng.uniform(0.0, 1.0, size=int(bootstrap_size)))
        freq, weights = bandpass_kresampling(
            h,
            nu_resampled,
            freq_range=(float(nu_ghz.min()), float(nu_ghz.max())),
            nresample=int(bootstrap_size),
        )
        weights = weights / trapezoid(weights, freq)
        centroid, bandwidth = compute_moments(freq, weights)
        out.append(
            {
                "frequency": freq,
                "weights": weights,
                "centroid": centroid,
                "bandwidth": bandwidth,
            }
        )
    return out

