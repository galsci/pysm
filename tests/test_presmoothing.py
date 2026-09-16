"""Tests for template presmoothing and differential smoothing.

These tests cover the ``SMOOTHING_ANGLE`` FITS header mechanism, the
``pre_applied_beam`` model attribute, the ``includes_smoothing`` property and the
``smoothing_angle`` build-time argument of :class:`pysm3.PowerLaw` and
:class:`pysm3.Sky`.
"""

import healpy as hp
import logging
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


def _band_limited_field(seed=0, nside=NSIDE):
    """Deterministic band-limited full-sky map with no power above ``LRAW``.

    The field has no monopole and no dipole (it starts at ``ell=2``): spin-2
    harmonics have no ``ell < 2`` modes, so any Q/U content at ``ell < 2``
    could not survive the TQU <-> TEB transforms used by the polarized
    smoothing path.
    """
    np.random.seed(seed)
    lmax_alm = int(2.5 * nside)
    alm = np.zeros(hp.Alm.getsize(lmax_alm), dtype=complex)
    for ell in range(2, LRAW + 1):
        for m in range(-ell, ell + 1):
            i = hp.Alm.getidx(lmax_alm, ell, abs(m))
            # scale power down with ell so the map is smooth
            alm[i] = (np.random.randn() + 1j * np.random.randn()) / (ell + 1.0) ** 1.5
    return hp.alm2map(alm, nside, lmax=lmax_alm)


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
    # a plain 0 (documented input) works too
    diff = get_differential_fwhm(1 * u.deg, 0)
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


def test_get_differential_beam_window_plain_zero_pre():
    """A plain 0 pre_applied_beam (documented input) works like in
    get_differential_fwhm, instead of raising AttributeError."""
    bw = get_differential_beam_window(1 * u.deg, 0, lmax=50)
    expected = hp.gauss_beam((1 * u.deg).to_value(u.rad), lmax=50)
    np.testing.assert_allclose(bw, np.tile(expected, (3, 1)))


@pytest.mark.parametrize("pre", [None, 0])
def test_get_differential_beam_window_per_component_zero_pre(pre):
    """Per-component targets remain per-component without presmoothing."""
    target = np.array([0.5, 0.5, 0.9]) * u.deg
    bw = get_differential_beam_window(target, pre, lmax=50)
    expected = np.stack(
        [hp.gauss_beam(fwhm.to_value(u.rad), lmax=50) for fwhm in target]
    )
    np.testing.assert_allclose(bw, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "bad_target",
    [np.array([0.5, 0.9]) * u.deg, np.array([0.5, 0.5, 0.5, 0.9]) * u.deg],
)
def test_get_differential_beam_window_invalid_target_size(bad_target):
    """A non scalar/(3,) target raises a clear ValueError, not an IndexError."""
    with pytest.raises(ValueError, match="target_fwhm"):
        get_differential_beam_window(bad_target, None, lmax=50)


def test_get_differential_beam_window_invalid_pre_size():
    """Invalid pre_applied_beam entry count raises a clear ValueError."""
    with pytest.raises(ValueError, match="pre_applied_beam"):
        get_differential_beam_window(1 * u.deg, np.array([0.5, 0.9]) * u.deg, lmax=50)


def test_get_differential_beam_window_no_underflow_nan():
    """Degree-scale beams at NSIDE-2048-scale lmax underflow both Gaussian
    windows to zero at high ell: the ratio must be exactly 0 there, not
    0/0 = NaN (and not the silently-wrong 0 the division gives where only
    the numerator underflows)."""
    lmax = 5120
    bw = get_differential_beam_window(2 * u.deg, 1.5 * u.deg, lmax=lmax)
    assert np.all(np.isfinite(bw))
    assert bw[0, -1] == 0.0  # underflowed to exactly zero
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = hp.gauss_beam((2 * u.deg).to_value(u.rad), lmax=lmax) / hp.gauss_beam(
            (1.5 * u.deg).to_value(u.rad), lmax=lmax
        )
    # documents the bug this guards against
    assert np.any(np.isnan(direct))
    # in the range where the direct division is valid, they agree
    valid = direct > 1e-30
    np.testing.assert_allclose(bw[0][valid], direct[valid], rtol=1e-12)


def test_apply_differential_smoothing_large_beams_no_nan():
    """End-to-end: large beams at NSIDE >= 256 must not produce NaN output
    maps from 0/0 window underflow in the per-component path."""
    nside = 256
    raw = _band_limited_field(seed=28, nside=nside)
    templ = np.stack([raw] * 3) * u.uK_RJ
    out = pysm3.apply_differential_smoothing(templ, np.ones(3) * u.deg, 5 * u.deg)
    assert np.all(np.isfinite(out.value))
    assert out.value.max() > 0


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


def test_extract_smoothing_angle_non_angle_value_is_zero(tmp_path):
    """A parseable but non-angle value (e.g. a plain number) is treated as no
    presmoothing instead of crashing every model built from the template."""
    raw = _band_limited_field(seed=20)
    path = tmp_path / "nonangle.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    add_metadata([path], field=1, smoothing_angle=53.0)  # number, not an angle
    m = read_map(str(path), nside=NSIDE, field=0)
    assert u.Quantity(m.smoothing_angle).value == 0.0
    # and a model can still be built from the template
    model = pysm3.PowerLaw(
        str(path), "23 GHz", -3.0, NSIDE, has_polarization=False, unit_I="uK_RJ"
    )
    np.testing.assert_array_equal(model.pre_applied_beam.value, 0.0)


def test_extract_smoothing_angle_negative_value_is_zero(tmp_path):
    """A negative header value is malformed (there are no negative beams) and
    is treated as no presmoothing with a warning, like other invalid values."""
    raw = _band_limited_field(seed=30)
    path = tmp_path / "neg.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    add_metadata([path], field=1, smoothing_angle="-53 arcmin")
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
def test_apply_differential_smoothing_utility(tmp_path):
    """The exported helper smooths by the differential (and never de-smooths)."""
    raw = _band_limited_field(seed=19)
    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    templ = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=pre, lmax=LMAX)
    out = pysm3.apply_differential_smoothing(templ, pre, target)
    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(out.value, expected.value, atol=1e-5 * expected.value.max())
    # no de-smoothing when the target is below the presmoothing
    assert pysm3.apply_differential_smoothing(templ, 0.9 * u.deg, 0.5 * u.deg) is templ


def test_apply_differential_smoothing_utility_per_component():
    """The utility also accepts a shape-(3,) pre_applied_beam with a (3, npix)
    map, taking each component to the target by its own differential."""
    raw = _band_limited_field(seed=22)
    pre = np.array([0.5, 0.4, 0.4]) * u.deg
    target = 0.9 * u.deg
    templ = np.stack(
        [
            apply_smoothing_and_coord_transform(
                raw * u.uK_RJ, fwhm=pre[i], lmax=LMAX
            ).value
            for i in range(3)
        ]
    ) * u.uK_RJ
    out = pysm3.apply_differential_smoothing(templ, pre, target)
    # identical to applying the explicitly-built per-component net window
    net_window = get_differential_beam_window(target, pre, lmax=LMAX)
    ref = apply_smoothing_and_coord_transform(
        templ, beam_window=net_window, lmax=LMAX
    )
    np.testing.assert_allclose(out.value, ref.value, rtol=1e-12, atol=0)
    # and the temperature component reaches the target exactly. (Q/U cannot
    # be compared pixel-by-pixel at this tolerance: healpix's polarized
    # spherical harmonic transforms carry O(1e-3) quadrature artifacts, e.g.
    # at the polar cap pixels. The window identity above plus the window
    # checks in the other tests cover them.)
    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(
        out[0].value, expected.value, atol=1e-5 * expected.value.max()
    )
    # a component whose target is below its presmoothing gets a unity window
    pre_over = np.array([0.5, 1.2, 0.4]) * u.deg
    bw = get_differential_beam_window(target, pre_over, lmax=LMAX)
    np.testing.assert_allclose(bw[1], 1.0)


def test_apply_differential_smoothing_utility_shape_mismatch():
    """A per-component pre_applied_beam with a single-component map errors
    clearly instead of failing inside healpy."""
    raw = _band_limited_field(seed=23)
    with pytest.raises(ValueError, match="3 components"):
        pysm3.apply_differential_smoothing(
            raw * u.uK_RJ, np.array([0.5, 0.4, 0.4]) * u.deg, 0.9 * u.deg
        )


def test_powerlaw_differential_smoothing_mpi_not_implemented(tmp_path):
    """smoothing_angle with MPI-distributed maps raises a clear error at build
    time instead of failing deep in the serial smoothing of a partial map."""
    raw = _band_limited_field(seed=21)
    path = tmp_path / "mpi.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    map_dist = pysm3.MapDistribution(
        pixel_indices=np.arange(hp.nside2npix(NSIDE))
    )
    with pytest.raises(NotImplementedError, match="map_dist"):
        pysm3.PowerLaw(
            str(path),
            "23 GHz",
            -3.0,
            NSIDE,
            has_polarization=False,
            unit_I="uK_RJ",
            smoothing_angle=1 * u.deg,
            map_dist=map_dist,
        )
    # the same guard applies when calling the method directly
    model = pysm3.PowerLaw(
        str(path),
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=False,
        unit_I="uK_RJ",
        map_dist=map_dist,
    )
    with pytest.raises(NotImplementedError, match="map_dist"):
        model.apply_differential_smoothing(1 * u.deg)


def test_model_extension_hook_is_noop():
    """Base Model.apply_differential_smoothing is a documented no-op hook."""
    m = pysm3.Model(nside=8)
    assert m.apply_differential_smoothing(1 * u.deg) is None


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
    """Different I vs Q/U presmoothing, each must reach the target, with Q/U
    smoothed through the joint TEB transform (PySM's IQU convention)."""
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

    # expected: a single joint (TEB) target smoothing of the raw maps
    raw_stack = np.stack([raw_I, raw_Q, raw_U]) * u.uK_RJ
    expected = apply_smoothing_and_coord_transform(raw_stack, fwhm=target, lmax=LMAX)
    # I is exact
    np.testing.assert_allclose(
        out[0].value,
        expected[0].value,
        atol=1e-5 * expected[0].value.max(),
    )
    # Q/U: the joint transform carries healpix SHT quadrature artifacts
    # (worst at the polar cap pixels), compare away from the poles with a
    # tolerance well above them. Smoothing Q/U as scalar maps instead would
    # exceed this tolerance (see test_differential_smoothing_preserves_pure_e)
    theta = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))[0]
    interior = (theta > 0.2) & (theta < np.pi - 0.2)
    for i in (1, 2):
        np.testing.assert_allclose(
            out[i].value[interior],
            expected[i].value[interior],
            atol=2e-3 * expected[i].value.max(),
        )


def test_differential_smoothing_preserves_pure_e(tmp_path):
    """Polarized differential smoothing uses the joint TEB transform: a pure-E
    template stays pure-E. Scalar (spin-0) smoothing of Q/U would mix E into
    B at a level orders of magnitude above the healpix SHT artifacts floor."""
    # build a pure-E band-limited field (no T, no B)
    np.random.seed(29)
    e_alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    for ell in range(2, LRAW + 1):
        for m in range(-ell, ell + 1):
            i = hp.Alm.getidx(LMAX, ell, abs(m))
            e_alm[i] = (np.random.randn() + 1j * np.random.randn()) / (ell + 1.0) ** 1.5
    zeros_alm = np.zeros_like(e_alm)
    tqu = np.array(
        hp.alm2map([zeros_alm, e_alm, zeros_alm], NSIDE, lmax=LMAX, pixwin=False)
    )

    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    # presmooth jointly, so the template really is pure-E
    path, _ = _presmoothed_template(tmp_path, "puree.fits", tqu, pre)
    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=True,
        freq_ref_P="23 GHz",
        smoothing_angle=target,
    )
    out = model.get_emission(23 * u.GHz).value

    alm_out = hp.map2alm(out, lmax=LMAX)
    e_level = np.abs(alm_out[1]).max()
    b_level = np.abs(alm_out[2]).max()
    # the B/E ratio stays at the SHT-artifacts floor (~1e-5); scalar
    # smoothing of Q/U mixes E into B at ~1e-3, 2 orders of magnitude higher
    assert b_level / e_level < 1e-4


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


def test_sky_applies_smoothing_angle_to_component_objects(tmp_path):
    """Sky applies smoothing_angle to pre-built component_objects through the
    apply_differential_smoothing hook instead of silently ignoring it."""
    raw = _band_limited_field(seed=25)
    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    path, _ = _presmoothed_template(tmp_path, "obj.fits", raw, pre)
    model = pysm3.PowerLaw(
        path, "23 GHz", -3.0, NSIDE, has_polarization=False, unit_I="uK_RJ"
    )
    sky = pysm3.Sky(component_objects=[model], nside=NSIDE, smoothing_angle=target)
    out = sky.get_emission(23 * u.GHz)[0]
    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(out.value, expected.value, atol=1e-5 * expected.value.max())


def test_sky_warns_on_unsupported_component_object(caplog):
    """A pre-built component without presmoothing support logs the same
    warning as config-built ones."""
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        pysm3.Sky(
            component_objects=[pysm3.Model(nside=8)],
            nside=8,
            smoothing_angle=1 * u.deg,
        )
    assert "Model does not support smoothing_angle" in caplog.text


def test_sky_forwards_smoothing_angle_to_nested_sky(tmp_path):
    """A Sky nested in component_objects forwards smoothing_angle to its own
    components through the hook."""
    raw = _band_limited_field(seed=26)
    pre = 0.5 * u.deg
    target = 0.9 * u.deg
    path, _ = _presmoothed_template(tmp_path, "nested.fits", raw, pre)
    inner = pysm3.Sky(
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
    outer = pysm3.Sky(
        component_objects=[inner], nside=NSIDE, smoothing_angle=target
    )
    out = outer.get_emission(23 * u.GHz)[0]
    expected = apply_smoothing_and_coord_transform(raw * u.uK_RJ, fwhm=target, lmax=LMAX)
    np.testing.assert_allclose(out.value, expected.value, atol=1e-5 * expected.value.max())


def test_differential_smoothing_is_idempotent(tmp_path):
    """pre_applied_beam records the applied target, so a second application
    (e.g. constructor and then the Sky hook) does not double-smooth."""
    raw = _band_limited_field(seed=27)
    path, _ = _presmoothed_template(tmp_path, "idem.fits", raw, 0.5 * u.deg)
    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=False,
        smoothing_angle=0.9 * u.deg,
        unit_I="uK_RJ",
    )
    out_first = model.get_emission(23 * u.GHz)[0].value.copy()
    np.testing.assert_allclose(
        model.pre_applied_beam.to_value(u.deg), [0.9, 0.9, 0.9], atol=1e-12
    )
    model.apply_differential_smoothing(0.9 * u.deg)  # second application: no-op
    np.testing.assert_allclose(
        model.get_emission(23 * u.GHz)[0].value, out_first, rtol=1e-12
    )


def test_smoothing_angle_accepts_strings(tmp_path):
    """smoothing_angle accepts strings like every other angle parameter, so
    TOML presets can use it (a Quantity cannot be written in TOML)."""
    raw = _band_limited_field(seed=32)
    path, _ = _presmoothed_template(tmp_path, "str.fits", raw, 0.5 * u.deg)
    expected = apply_smoothing_and_coord_transform(
        raw * u.uK_RJ, fwhm=0.9 * u.deg, lmax=LMAX
    )
    # directly on the model
    model = pysm3.PowerLaw(
        path,
        "23 GHz",
        -3.0,
        NSIDE,
        has_polarization=False,
        smoothing_angle="0.9 deg",
        unit_I="uK_RJ",
    )
    np.testing.assert_allclose(
        model.get_emission(23 * u.GHz)[0].value,
        expected.value,
        atol=1e-5 * expected.value.max(),
    )
    # through Sky
    sky = pysm3.Sky(
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
        smoothing_angle="0.9 deg",
    )
    assert u.Quantity(sky.smoothing_angle).to_value(u.deg) == pytest.approx(0.9)
    np.testing.assert_allclose(
        sky.get_emission(23 * u.GHz)[0].value,
        expected.value,
        atol=1e-5 * expected.value.max(),
    )
    # as a component-configuration value with no Sky-level override
    sky_cfg = pysm3.Sky(
        component_config={
            "mys1": {
                "class": "PowerLaw",
                "map_I": str(path),
                "freq_ref_I": "23 GHz",
                "map_pl_index": -3.0,
                "has_polarization": False,
                "unit_I": "uK_RJ",
                "smoothing_angle": "0.9 deg",
            }
        },
        nside=NSIDE,
    )
    np.testing.assert_allclose(
        sky_cfg.get_emission(23 * u.GHz)[0].value,
        expected.value,
        atol=1e-5 * expected.value.max(),
    )


def test_sky_warns_on_unsupported_component(tmp_path, caplog):
    """A smoothing_angle requested on a sky that has components without
    presmoothing support logs a warning (their emission is left unsmoothed)."""
    raw = _band_limited_field(seed=24)
    path = tmp_path / "plain.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        pysm3.Sky(
            component_config={
                "base": {"class": "Model"},
                "mys1": {
                    "class": "PowerLaw",
                    "map_I": str(path),
                    "freq_ref_I": "23 GHz",
                    "map_pl_index": -3.0,
                    "has_polarization": False,
                    "unit_I": "uK_RJ",
                },
            },
            nside=NSIDE,
            smoothing_angle=1 * u.deg,
        )
    assert "Model does not support smoothing_angle" in caplog.text
    # no warning when no target beam is requested
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        pysm3.Sky(
            component_config={"base": {"class": "Model"}},
            nside=NSIDE,
        )
    assert caplog.text == ""


def test_sky_warns_when_overriding_component_smoothing_angle(tmp_path, caplog):
    """A Sky-level smoothing_angle takes precedence over a smoothing_angle
    key in a component's configuration, with a warning."""
    raw = _band_limited_field(seed=31)
    path = tmp_path / "plain.fits"
    hp.write_map(path, raw, dtype=np.float64, overwrite=True)
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        sky = pysm3.Sky(
            component_config={
                "mys1": {
                    "class": "PowerLaw",
                    "map_I": str(path),
                    "freq_ref_I": "23 GHz",
                    "map_pl_index": -3.0,
                    "has_polarization": False,
                    "unit_I": "uK_RJ",
                    "smoothing_angle": "0.5 deg",  # overridden by Sky-level
                }
            },
            nside=NSIDE,
            smoothing_angle=0.9 * u.deg,
        )
    assert "Ignoring the smoothing_angle value" in caplog.text
    # the Sky-level target is the one applied
    np.testing.assert_allclose(
        sky.components[0].pre_applied_beam.to_value(u.deg),
        [0.9, 0.9, 0.9],
        atol=1e-12,
    )


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
