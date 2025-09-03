# Important: it is important to import netCDF4 before `h4py` is imported
# to avoid "HDF Error" under Ubuntu with pip
# This does not happen with conda packages
import h5py
import healpy as hp
import netCDF4
import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pysm3 import units as u
from pysm3 import utils
from pysm3.models.catalog import (
    PointSourceCatalog,
    evaluate_model,
    evaluate_poly,
    aggregate,
)
from pysm3.utils import car_aperture_photometry, healpix_aperture_photometry


def test_aggregate():
    m = np.zeros(3)
    aggregate(np.array([2, 2]), m, np.ones(2))
    assert_allclose(m, np.array([0, 0, 2]))


def test_evaluate_poly():
    np.random.seed(100)
    for N in [4, 5, 6]:
        p = np.random.rand(N, 1)
        x = np.random.rand(1)[0]
        assert_allclose(np.polyval(p.flatten(), x), evaluate_poly(p, x)[0])


def test_evaluate_model_1freq_flat():
    coeff = np.array([[0, 0, 0, 0, 3.7]]).T  # Transpose to (n_coeff, 1)
    freqs = np.array([100])
    weights = np.array([0])  # not used when 1 point
    assert np.allclose(evaluate_model(freqs, weights, coeff), np.array([3.7]))


def test_evaluate_model_1freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]]).T  # Transpose to (n_coeff, 1)
    freqs = np.array([np.exp(3)])
    weights = np.array([0])
    assert np.allclose(evaluate_model(freqs, weights, coeff), np.array([6.0]))


def test_evaluate_model_2freq_flat():
    coeff = np.array([[0, 0, 0, 0, 3.7]]).T
    freqs = np.exp(np.array([3, 4]))  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= trapezoid(weights, x=freqs)
    assert_allclose(evaluate_model(freqs, weights, coeff), np.array([3.7]))


def test_evaluate_model_2freq_lin():
    coeff = np.array([[0, 0, 0, 2, 0]]).T
    freqs = np.exp(np.array([3, 4]))  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= trapezoid(weights, x=freqs)
    flux = evaluate_model(freqs, weights, coeff)[0]
    assert flux > 6
    assert flux < 8
    assert_allclose(flux, trapezoid(weights * np.array([6, 8]), x=freqs))


@pytest.fixture(scope="session")
def test_catalog(tmp_path_factory):
    num_sources = 2
    indices = np.arange(num_sources)
    num_powers = 5  # Assuming degree 4, so 5 coefficients

    dims = ("power", "index")  # Changed dimension order
    catalog = xr.Dataset(
        {
            "logpolycoefflux": (
                dims,
                np.zeros((num_powers, len(indices)), dtype=np.float64),
            ),  # Adjusted shape
            "logpolycoefpolflux": (
                dims,
                np.zeros((num_powers, len(indices)), dtype=np.float64),
            ),  # Adjusted shape
        },
        coords={
            "index": indices,
            "power": np.arange(num_powers)[::-1],  # Still highest degree first
            "theta": ("index", np.array([np.pi / 4, np.pi / 2])),
            "phi": ("index", np.zeros(num_sources)),
        },
    )
    for field in ["theta", "phi"]:
        catalog[field].attrs["units"] = "rad"
    for field in ["logpolycoefflux", "logpolycoefpolflux"]:
        catalog[field].attrs["units"] = "Jy"
    catalog.attrs["description"] = "Test catalog"
    catalog.attrs["reference_frequency_GHz"] = 100.0
    catalog.attrs["flux_cutoff_mJy"] = 1.0
    catalog.attrs["polynomial_degree"] = 4
    catalog.attrs[
        "sorted_by"
    ] = "polyval(logpolycoefflux, log(ref_freq)) with coeff dims (power,index)"
    catalog.attrs["ref_frame"] = "Galactic"
    catalog.attrs["generated_utc"] = "2025-09-02T12:00:00.000000+00:00"
    catalog.attrs["git_commit"] = "test_commit_hash"
    catalog.attrs["command"] = "test_command"
    # Adjust loc indexing to match new dimension order (power, index)
    catalog["logpolycoefflux"].loc[
        {"power": 0, "index": 0}
    ] = 3.7  # power 0 is x^0 (constant term)
    catalog["logpolycoefflux"].loc[
        {"power": 1, "index": 1}
    ] = 2  # power 1 is x^1 (linear term)
    catalog["logpolycoefpolflux"].loc[
        {"power": 0, "index": 1}
    ] = 5  # power 0 is x^0 (constant term)
    fn = tmp_path_factory.mktemp("data") / "test_catalog.h5"
    print(netCDF4.__version__)
    catalog.to_netcdf(str(fn), format="NETCDF4")  # requires netcdf4 package
    return str(fn)


def test_catalog_class_fluxes(test_catalog):
    nside = 8
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    normalized_weights = utils.normalize_weights(utils.check_freq_input(freqs), weights)
    flux = catalog.get_fluxes(freqs, weights=weights)
    assert_allclose(flux[0], 3.7 * u.Jy)
    assert (
        flux[1]
        == trapezoid(normalized_weights * np.array([6, 8]), x=freqs.to_value(u.GHz))
        * u.Jy
    )


def test_catalog_class_map_no_beam(test_catalog):
    nside = 8
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= trapezoid(weights, x=freqs.to_value(u.GHz))

    scaling_factor = utils.bandpass_unit_conversion(
        freqs, weights, output_unit=u.uK_RJ, input_unit=u.Jy / u.sr
    ) / (hp.nside2pixarea(nside) * u.sr)
    flux_I = catalog.get_fluxes(freqs, weights=weights)

    flux_P = catalog.get_fluxes(freqs, weights=weights, coeff="logpolycoefpolflux")
    output_map = catalog.get_emission(freqs, weights=weights, fwhm=None)
    with h5py.File(test_catalog) as f:
        pix = hp.ang2pix(nside, f["theta"], f["phi"])
    assert_allclose(
        output_map[0, pix] / scaling_factor,  # convert to flux
        flux_I,
    )
    np.random.seed(56567)
    psirand = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=len(flux_P))
    assert_allclose(output_map[1, pix] / scaling_factor, flux_P * np.cos(2 * psirand))
    assert_allclose(output_map[2, pix] / scaling_factor, flux_P * np.sin(2 * psirand))


def test_catalog_class_map_beam(test_catalog):
    # resolution of the map is 7 degrees, beam is 0.5 degrees
    # all flux should be in the central pixel
    nside = 32
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= trapezoid(weights, x=freqs.to_value(u.GHz))

    scaling_factor = utils.bandpass_unit_conversion(
        freqs, weights, input_unit=u.uK_RJ, output_unit=u.Jy / u.sr
    )
    catalog_flux = catalog.get_fluxes(freqs, weights=weights)

    fwhm = 2 * u.deg
    output_map = catalog.get_emission(
        freqs,
        weights=weights,
        fwhm=fwhm,
        return_car=True,
        return_healpix=False,
    )
    assert_allclose(
        output_map[0].argmax(unit="coord"), np.array([0, 0]), atol=1e-2, rtol=1e-3
    )

    box_half_size_rad = 3 * fwhm.to_value(u.rad)
    box_center = [0, 0]
    box = np.array(
        [
            [box_center[0] - box_half_size_rad, box_center[1] - box_half_size_rad],
            [box_center[0] + box_half_size_rad, box_center[1] + box_half_size_rad],
        ]
    )  # in radians
    cutout = output_map[0].submap(box) * scaling_factor.value
    flux = car_aperture_photometry(cutout, 2 * fwhm.to_value(u.rad)) * u.Jy
    assert_allclose(flux, catalog_flux.max(), rtol=1e-3)

    catalog_flux_P = catalog.get_fluxes(
        freqs, weights=weights, coeff="logpolycoefpolflux"
    )
    np.random.seed(56567)
    psirand = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=2)
    cutout = output_map[1].submap(box) * scaling_factor.value
    flux = car_aperture_photometry(cutout, 2 * fwhm.to_value(u.rad)) * u.Jy
    assert_allclose(flux, catalog_flux_P[1] * np.cos(2 * psirand[1]), rtol=1e-3)

    cutout = output_map[2].submap(box) * scaling_factor.value
    flux = car_aperture_photometry(cutout, 2 * fwhm.to_value(u.rad)) * u.Jy
    assert_allclose(flux, catalog_flux_P[1] * np.sin(2 * psirand[1]), rtol=1e-3)


def test_catalog_class_map_healpix(test_catalog):
    # resolution of the map is 7 degrees, beam is 0.5 degrees
    # all flux should be in the central pixel
    nside = 32
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= trapezoid(weights, x=freqs.to_value(u.GHz))

    scaling_factor = utils.bandpass_unit_conversion(
        freqs, weights, input_unit=u.uK_RJ, output_unit=u.Jy / u.sr
    )
    catalog_flux = catalog.get_fluxes(freqs, weights=weights)

    fwhm = 2 * u.deg
    output_map = catalog.get_emission(
        freqs,
        weights=weights,
        fwhm=fwhm,
        return_car=False,
    )
    with h5py.File(test_catalog) as f:
        pix = hp.ang2pix(nside, f["theta"], f["phi"])
        theta = np.array(f["theta"])
        phi = np.array(f["phi"])
    assert output_map.argmax() == pix[1]

    flux = healpix_aperture_photometry(
        (output_map[0].value * scaling_factor.value),
        aperture_radius=2 * fwhm.to_value(u.rad),
        theta=theta[1],
        phi=phi[1],
    )
    assert_allclose(flux, catalog_flux.max().value, rtol=4e-2)  # loose 4%


def test_catalog_slicing_get_fluxes(test_catalog):
    nside = 8  # Add nside
    # Test with a slice that selects the second source (index 1)
    catalog = PointSourceCatalog(test_catalog, nside=nside, catalog_slice=slice(1, 2))
    freqs = np.exp(np.array([3, 4])) * u.GHz
    weights = np.array([1, 1], dtype=np.float64)
    flux = catalog.get_fluxes(freqs, weights=weights)

    # The second source has a linear term (coeff 2 for power 1)
    # For logfreqs [3, 4], the fluxes should be [2*3, 2*4] = [6, 8]
    # After bandpass integration, it should be the same as test_evaluate_model_2freq_lin
    normalized_weights = utils.normalize_weights(utils.check_freq_input(freqs), weights)
    expected_flux = (
        trapezoid(normalized_weights * np.array([6, 8]), x=freqs.to_value(u.GHz)) * u.Jy
    )

    assert_allclose(flux[0], expected_flux)

    # Test with a slice that selects the first source (index 0)
    catalog = PointSourceCatalog(test_catalog, nside=nside, catalog_slice=slice(0, 1))
    freqs = np.exp(np.array([3, 4])) * u.GHz
    weights = np.array([1, 1], dtype=np.float64)
    flux = catalog.get_fluxes(freqs, weights=weights)

    # The first source has a constant term (coeff 3.7 for power 0)
    # After bandpass integration, it should be 3.7
    assert_allclose(flux[0], 3.7 * u.Jy)
