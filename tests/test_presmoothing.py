"""Tests for template presmoothing and differential smoothing.

These tests cover the ``SMOOTHING_ANGLE`` FITS header mechanism, the
``pre_applied_beam`` model attribute, the ``includes_smoothing`` property and the
``smoothing_angle`` build-time argument of :class:`pysm3.PowerLaw` and
:class:`pysm3.Sky`.
"""

import healpy as hp
import numpy as np
import pytest

import pysm3
import pysm3.units as u
from pysm3 import (
    apply_smoothing_and_coord_transform,
    get_differential_beam_window,
    get_differential_fwhm,
    read_map,
)
from pysm3.utils.add_metadata import add_metadata

NSIDE = 64
LMAX = int(2.5 * NSIDE)
# band-limit the raw field so harmonic smoothing is exact and no pixel-window
# edge effects leak into the comparisons
LRAW = 20


def _band_limited_field(seed=0):
    """Deterministic band-limited full-sky map with no power above ``LRAW``."""
    np.random.seed(seed)
    lmax_alm = LMAX
    alm = np.zeros(hp.Alm.getsize(lmax_alm), dtype=complex)
    for ell in range(LRAW + 1):
        for m in range(-ell, ell + 1):
            i = hp.Alm.getidx(lmax_alm, ell, abs(m))
            # scale power down with ell so the map is smooth
            alm[i] = (np.random.randn() + 1j * np.random.randn()) / (ell + 1.0) ** 1.5
    alm[0] += 0.0  # l=0 term zero
    return hp.alm2map(alm, NSIDE, lmax=lmax_alm)



def _presmoothed_template(tmp_path, name, raw, fwhm):
    """Write a template made by smoothing `raw` by `fwhm`, tagging its header."""
    if raw.ndim == 1:
        raw3 = np.zeros((3, raw.size))
        raw3[0] = raw
        raw = raw3
    smoothed = apply_smoothing_and_coord_transform(
        raw * u.uK_RJ, fwhm=fwhm, lmax=LMAX
    )
    path = tmp_path / name
    hp.write_map(path, smoothed.value, dtype=np.float64, overwrite=True)
    add_metadata(
        [path],
        field=1,
        unit="uK_RJ",
        smoothing_angle=f"{fwhm.to_value(u.deg):.6f} deg",
    )
    return str(path), smoothed


# --------------------------------------------------------------------------- #
# get_differential_fwhm
# --------------------------------------------------------------------------- #
def test_get_differential_fwhm_scalar():
    # sqrt(1 deg**2 - 0.7 deg**2)
    diff = get_differential_fwhm(1 * u.deg, 0.7 * u.deg)
    expected = np.sqrt(1.0**2 - 0.7**2) * u.deg
    assert u.Quantity(diff).unit == u.Unit(u.rad)
    assert diff.value == pytest.approx(expected.to_value(u.rad), rel=1e-12)


def test_get_differential_fwhm_no_presmoothing():
    # no pre-applied beam -> need the full target
    diff = get_differential_fwhm(1 * u.deg)
    assert diff.value == pytest.approx((1 * u.deg).to_value(u.rad))
    diff = get_differential_fwhm(1 * u.deg, 0 * u.deg)
    assert diff.value == pytest.approx((1 * u.deg).to_value(u.rad))


def test_get_differential_fwhm_target_le_presmoothing_identity():
    # cannot de-convolve: differential must be 0
    assert get_differential_fwhm(0.5 * u.deg, 0.7 * u.deg).value == 0.0
    assert get_differential_fwhm(0.7 * u.deg, 0.7 * u.deg).value == 0.0


def test_get_differential_fwhm_per_component():
    # scalar target, per-component presmoothing -> per-component differential
    diff = get_differential_fwhm(1 * u.deg, np.array([0.7, 0.5, 0.5]) * u.deg)
    assert diff.shape == (3,)
    expected = np.sqrt(1.0**2 - np.array([0.7, 0.5, 0.5]) ** 2)
    np.testing.assert_allclose(
        diff.to_value(u.deg), expected, rtol=1e-12
    )


def test_get_differential_fwhm_quadrature_smoking():
    # window-division and fwhm-quadrature agree for Gaussian beams
    for R, pre in [(1 * u.deg, 0.7 * u.deg), (0.9 * u.deg, 0.4 * u.deg)]:
        diff_fwhm = get_differential_fwhm(R, pre)
        bw = get_differential_beam_window(R, pre, lmax=100)
        gauss_diff = hp.gauss_beam(diff_fwhm.to_value(u.rad), lmax=100)
        np.testing.assert_allclose(bw[0], gauss_diff, rtol=1e-12)


# --------------------------------------------------------------------------- #
# get_differential_beam_window (window-division convention)
# --------------------------------------------------------------------------- #
def test_get_differential_beam_window_identity_when_le():
    # no de-smoothing when target <= pre: window is unity
    bw = get_differential_beam_window(0.5 * u.deg, 0.7 * u.deg, lmax=50)
    np.testing.assert_allclose(bw, 1.0)
    np.testing.assert_allclose(
        get_differential_beam_window(0.5 * u.deg, 0.7 * u.deg, lmax=50), 1.0
    )


def test_get_differential_beam_window_per_component():
    bw = get_differential_beam_window(
        1 * u.deg, np.array([0.7, 0.5, 0.5]) * u.deg, lmax=50
    )
    assert bw.shape == (3, 51)
    # I row (pre 0.7 deg) must be *less* attenuated than Q/U rows (pre 0.5 deg)
    assert np.all(bw[0, 5:] >= bw[1, 5:])


def test_get_differential_beam_window_matches_direct_division():
    bw = get_differential_beam_window(1 * u.deg, 0.7 * u.deg, lmax=50)
    direct = hp.gauss_beam((1 * u.deg).to_value(u.rad), lmax=50) / hp.gauss_beam(
        (0.7 * u.deg).to_value(u.rad), lmax=50
    )
    np.testing.assert_allclose(bw[0], direct, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Header parsing (SMOOTHING_ANGLE round trip)
# --------------------------------------------------------------------------- #
def test_read_map_records_smoothing_angle(tmp_path):
    raw = _band_limited_field(seed=2)
    path, _ = _presmoothed_template(tmp_path, "smoothed.fits", raw, 53 * u.arcmin)
    m = read_map(path, nside=NSIDE, field=[0, 1, 2], unit="uK_RJ")
    assert u.Quantity(m.smoothing_angle).to(u.arcmin).value == pytest.approx(
        53.0, rel=1e-6
    )


def test_read_map_no_smoothing_key(tmp_path):
    raw = _band_limited_field(seed=3)
    path = tmp_path / "plain.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    m = read_map(str(path), nside=NSIDE, field=0)
    assert u.Quantity(m.smoothing_angle).value == 0.0


def test_extract_smoothing_angle_bad_value_is_zero(tmp_path):
    raw = _band_limited_field(seed=4)
    path = tmp_path / "bad.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    add_metadata([path], field=1, smoothing_angle="not-an-angle")
    m = read_map(str(path), nside=NSIDE, field=0)
    assert u.Quantity(m.smoothing_angle).value == 0.0


def test_powerlaw_pre_applied_beam(tmp_path):
    raw = _band_limited_field(seed=5)
    path, _ = _presmoothed_template(tmp_path, "i.fits", raw, 56 * u.arcmin)
    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=True,  # IQU template, presmoothing applies to all
    )
    # 56 arcmin recorded for I, Q and U
    np.testing.assert_allclose(
        model.pre_applied_beam.to_value(u.arcmin), [56.0, 56.0, 56.0], atol=1e-4
    )
    assert model.includes_smoothing


def test_powerlaw_no_presmoothing(tmp_path):
    raw = _band_limited_field(seed=6)
    path = tmp_path / "plain.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)  # no header key
    model = pysm3.PowerLaw(
        str(path), "23 GHz", -3.0, NSIDE, has_polarization=False, unit_I="uK_RJ"
    )
    np.testing.assert_array_equal(model.pre_applied_beam.value, 0.0)
    assert not model.includes_smoothing


def test_powerlaw_separate_I_QU_presmoothing(tmp_path):
    """The motivating case: I and Q/U templates carry different beams."""
    raw_I = _band_limited_field(seed=7)
    raw_Q = _band_limited_field(seed=8)
    raw_U = _band_limited_field(seed=9)
    path_i, _ = _presmoothed_template(tmp_path, "i.fits", raw_I, 56 * u.arcmin)
    path_q, _ = _presmoothed_template(tmp_path, "q.fits", raw_Q, 53 * u.arcmin)
    path_u, _ = _presmoothed_template(tmp_path, "u.fits", raw_U, 53 * u.arcmin)
    model = pysm3.PowerLaw(
        path_i,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=True,
        map_Q=path_q,
        map_U=path_u,
        freq_ref_P="23 GHz",
    )
    np.testing.assert_allclose(
        model.pre_applied_beam.to_value(u.arcmin),
        [56.0, 53.0, 53.0],
        atol=1e-4,
    )
    assert model.includes_smoothing


# --------------------------------------------------------------------------- #
# End-to-end differential smoothing
# --------------------------------------------------------------------------- #
def test_differential_smoothing_end_to_end(tmp_path):
    raw = _band_limited_field(seed=10)
    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    path, _ = _presmoothed_template(tmp_path, "e2e.fits", raw, pre)

    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=False,
        smoothing_angle=target,
        unit_I="uK_RJ",
    )
    out = model.get_emission(23 * u.GHz)[0]

    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    # differential smoothing reproduces a single full target smoothing
    np.testing.assert_allclose(
        out.value, expected.value, atol=1e-5 * expected.value.max()
    )

    # ... and is far closer than naively smoothing the pre-smoothed template
    # by the full target (which double-applies part of the beam)
    naive = apply_smoothing_and_coord_transform(
        apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=pre, lmax=LMAX),
        fwhm=target,
        lmax=LMAX,
    )
    assert (
        np.abs(out - expected).value.max()
        < 0.5 * np.abs(naive - expected).value.max()
    )


def test_differential_smoothing_per_component(tmp_path):
    """Different I vs Q/U presmoothing, each must reach the target exactly."""
    raw_I = _band_limited_field(seed=11)
    raw_Q = _band_limited_field(seed=12)
    raw_U = _band_limited_field(seed=13)
    path_i, _ = _presmoothed_template(tmp_path, "pi.fits", raw_I, 0.5 * u.deg)
    path_q, _ = _presmoothed_template(tmp_path, "pq.fits", raw_Q, 0.4 * u.deg)
    path_u, _ = _presmoothed_template(tmp_path, "pu.fits", raw_U, 0.4 * u.deg)
    target = 0.9 * u.deg

    model = pysm3.PowerLaw(
        path_i,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=True,
        map_Q=path_q,
        map_U=path_u,
        freq_ref_P="23 GHz",
        smoothing_angle=target,
    )
    out = model.get_emission(23 * u.GHz)

    exp_I = apply_smoothing_and_coord_transform(raw_I * u.uK_RJ, fwhm=target, lmax=LMAX)
    exp_Q = apply_smoothing_and_coord_transform(raw_Q * u.uK_RJ, fwhm=target, lmax=LMAX)
    exp_U = apply_smoothing_and_coord_transform(raw_U * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(out[0].value, exp_I.value, atol=1e-5 * exp_I.value.max())
    np.testing.assert_allclose(out[1].value, exp_Q.value, atol=1e-5 * exp_Q.value.max())
    np.testing.assert_allclose(out[2].value, exp_U.value, atol=1e-5 * exp_U.value.max())


def test_smoothing_angle_smaller_than_presmoothing_is_noop(tmp_path):
    """If the target is below the template's presmoothing, do not de-smooth."""
    raw = _band_limited_field(seed=14)
    pre = 0.9 * u.deg
    target = 0.5 * u.deg  # smaller than pre -> cannot de-convolve
    path, template_smoothed = _presmoothed_template(tmp_path, "lores.fits", raw, pre)
    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=False,
        smoothing_angle=target,
        unit_I="uK_RJ",
    )
    out = model.get_emission(23 * u.GHz)[0]
    # output should be unchanged w.r.t. the (already over-resolved) template:
    # the I component of the template
    template_smoothed_i = (
        template_smoothed[0] if template_smoothed.shape[0] == 3 else template_smoothed
    )
    np.testing.assert_allclose(out.value, template_smoothed_i.value, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Sky-level forwarding & backwards compatibility
# --------------------------------------------------------------------------- #
def test_sky_forwards_smoothing_angle(tmp_path):
    raw = _band_limited_field(seed=15)
    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    path, _ = _presmoothed_template(tmp_path, "sky.fits", raw, pre)
    config = {
        "mys1": {
            "class": "PowerLaw",
            "map_I": str(path),
            "freq_ref_I": "23 GHz",
            "map_pl_index": -3.0,
            "has_polarization": False,
            "unit_I": "uK_RJ",
        }
    }
    sky = pysm3.Sky(component_config=config, nside=NSIDE, smoothing_angle=target)
    assert sky.smoothing_angle == target
    out = sky.get_emission(23 * u.GHz)[0]
    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(out.value, expected.value, atol=1e-5 * expected.value.max())


def test_sky_without_smoothing_angle_backwards_compatible(tmp_path):
    """No smoothing_angle, no header -> unchanged emission (no smoothing)."""
    raw = _band_limited_field(seed=16)
    path = tmp_path / "plain.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)  # no SMOOTHING_ANGLE
    model = pysm3.Sky(
        preset_strings=[],
        component_objects=[],
        component_config={
            "mys1": {
                "class": "PowerLaw",
                "map_I": str(path),
                "freq_ref_I": "23 GHz",
                "map_pl_index": -3.0,
                "has_polarization": False,
                "unit_I": "uK_RJ",
            }
        },
        nside=NSIDE,
    )
    out = model.get_emission(23 * u.GHz)[0]
    comp = model.components[0]
    assert not comp.includes_smoothing
    # no beam applied at all: emission == raw template (scaled by the unit
    # conversion factor of 1 at 23 GHz == freq_ref)
    np.testing.assert_allclose(out.value, raw, atol=1e-5 * np.abs(raw).max())


# --------------------------------------------------------------------------- #
# Per-component differential via the window-division beam_window path
# --------------------------------------------------------------------------- #
def test_differential_beam_window_apply_per_component():
    """A per-component differential beam_window applied through
    `apply_smoothing_and_coord_transform` reproduces the per-component target."""
    raw = _band_limited_field(seed=18)
    pre = np.array([0.5, 0.4, 0.4]) * u.deg
    target = 0.9 * u.deg
    # build a per-component pre-smoothed template (scalar fwhm per component)
    raw3 = np.zeros((3, raw.size))
    raw3[0] = raw
    templ = np.empty_like(raw3)
    for i in range(3):
        templ[i] = apply_smoothing_and_coord_transform(
            raw3[i] * u.uK_RJ, fwhm=pre[i], lmax=LMAX
        ).value
    templ = templ * u.uK_RJ
    net_window = get_differential_beam_window(target, pre, lmax=LMAX)
    out = apply_smoothing_and_coord_transform(
        templ, beam_window=net_window, lmax=LMAX
    )
    for i in range(3):
        expected_i = apply_smoothing_and_coord_transform(
            raw3[i] * u.uK_RJ, fwhm=target, lmax=LMAX
        )
        np.testing.assert_allclose(
            out[i].value, expected_i.value, atol=1e-5 * expected_i.value.max()
        )
