from pysm3 import units as u
from pysm3.utils import car_aperture_photometry
import h5py
import healpy as hp
from numpy.testing import assert_allclose
import pytest

from pysm3.models.catalog import evaluate_poly, evaluate_model, PointSourceCatalog
import numpy as np
import xarray as xr
from pixell import enmap

from pysm3 import utils


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
def test_catalog(tmp_path_factory):
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
            "theta": ("index", np.array([np.pi / 4, np.pi / 2])),
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


def test_catalog_class_fluxes(test_catalog):
    nside = 8
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs.to_value(u.GHz))
    flux = catalog.get_fluxes(freqs, weights=weights)
    assert_allclose(flux[0], 3.7 * u.Jy)
    assert (
        flux[1] == np.trapz(weights * np.array([6, 8]), x=freqs.to_value(u.GHz)) * u.Jy
    )


def test_catalog_class_map_no_beam(test_catalog):
    nside = 8
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs.to_value(u.GHz))

    scaling_factor = utils.bandpass_unit_conversion(
        freqs, weights, output_unit=u.uK_RJ, input_unit=u.Jy / u.sr
    ) / (hp.nside2pixarea(nside) * u.sr)
    flux_I = catalog.get_fluxes(freqs, weights=weights)

    flux_P = catalog.get_fluxes(freqs, weights=weights, coeff="logpolycoefpolflux")
    output_map = catalog.get_emission(
        freqs, weights=weights, output_units=u.uK_RJ, fwhm=None
    )
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


def test_catalog_class_map_small_beam(test_catalog):
    # resolution of the map is 7 degrees, beam is 0.5 degrees
    # all flux should be in the central pixel
    nside = 32
    catalog = PointSourceCatalog(test_catalog, nside=nside)
    freqs = np.exp(np.array([3, 4])) * u.GHz  # ~ 20 and ~ 55 GHz
    weights = np.array([1, 1], dtype=np.float64)
    weights /= np.trapz(weights, x=freqs.to_value(u.GHz))

    scaling_factor = utils.bandpass_unit_conversion(
        freqs, weights, output_unit=u.uK_RJ, input_unit=u.Jy / u.sr
    ) / (hp.nside2pixarea(nside) * u.sr)
    surface_brigthness = catalog.get_fluxes(freqs, weights=weights) * scaling_factor

    fwhm = 1 * u.deg
    output_map = catalog.get_emission(
        freqs, weights=weights, output_units=u.uK_RJ, fwhm=fwhm
    )
    with h5py.File(test_catalog) as f:
        pix = np.round(
            enmap.sky2pix(
                output_map.shape,
                output_map.wcs,
                np.column_stack((np.pi / 2 - np.array(f["theta"]), f["phi"])),
            )
        ).astype(int)
    assert_allclose(
        output_map[0].argmax(unit="coord"), np.array([0, 0]), atol=1e-2, rtol=1e-3
    )

    box_half_size_rad = 2 * fwhm.to_value(u.rad)
    box_center = [0, 0]
    box = np.array(
        [
            [box_center[0] - box_half_size_rad, box_center[1] - box_half_size_rad],
            [box_center[0] + box_half_size_rad, box_center[1] + box_half_size_rad],
        ]
    )  # in radians
    cutout = output_map[0].submap(box)
    flux = car_aperture_photometry(cutout, 2 * fwhm.to_value(u.rad))
    assert_allclose(flux, surface_brigthness[0].to_value(u.uK_RJ), rtol=1e-3)
