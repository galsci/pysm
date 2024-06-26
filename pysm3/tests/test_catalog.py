from astropy import units as u
from numpy.testing import assert_allclose
import pytest

from pysm3.models.catalog import evaluate_poly, evaluate_model, PointSourceCatalog
import numpy as np
import xarray as xr


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
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs)
    assert evaluate_model(freqs, weights, coeff) == np.ones((1, 1)) * 3.7


def test_evaluate_model_2freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]])
    freqs = np.exp(np.array([3, 4]))  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs)
    flux = evaluate_model(freqs, weights, coeff)[0]
    assert flux > 6
    assert flux < 8
    assert flux == np.trapz(weights * np.array([6, 8]), x=freqs)


@pytest.fixture(scope="session")
def generate_test_catalog(tmp_path_factory):
    num_sources = 2
    indices = np.arange(num_sources)

    dims = ("index", "power")
    catalog = xr.Dataset(
        {
            "logpolycoefflux": (dims, np.zeros((len(indices), 5), dtype=np.float64)),
            "logpolycoefpolflux": (dims, np.zeros((len(indices), 5), dtype=np.float64)),
        },
        coords={
            "index": indices,
            "power": np.arange(5)[::-1],
            "theta": ("index", np.zeros(num_sources)),
            "phi": ("index", np.zeros(num_sources)),
        },
    )
    for field in ["theta", "phi"]:
        catalog[field].attrs["units"] = "rad"
    for field in ["logpolycoefflux", "logpolycoefpolflux"]:
        catalog[field].attrs["units"] = "Jy"
    catalog["logpolycoefflux"].loc[dict(index=0, power=0)] = 3.7
    catalog["logpolycoefflux"].loc[dict(index=1, power=1)] = 2
    fn = tmp_path_factory.mktemp("data") / "test_catalog.h5"
    catalog.to_netcdf(str(fn), format="NETCDF4")  # requires netcdf4 package
    return str(fn)


def test_catalog_class(generate_test_catalog):
    nside = 64
    catalog = PointSourceCatalog(generate_test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs.to_value(u.GHz))
    flux = catalog.get_fluxes(freqs, weights=weights)
    assert_allclose(flux[0], 3.7)
    assert flux[1] == np.trapz(weights * np.array([6, 8]), x=freqs)
