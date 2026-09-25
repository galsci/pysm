"""Test automatic differential smoothing from template-declared pre-applied beam

Implements the test plan of https://github.com/galsci/pysm/issues/272
"""

import logging

import healpy as hp
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.units import UnitConversionError

import pysm3
import pysm3.units as u
from pysm3 import apply_smoothing_and_coord_transform, get_differential_fwhm, map2alm
from pysm3.sky import get_common_pre_applied_fwhm

NSIDE = 64
LMAX = int(1.5 * NSIDE)
PRE_FWHM = 56 * u.arcmin
TARGET_FWHM = 1 * u.deg
LARGER_FWHM = 2 * u.deg


@pytest.fixture(scope="module")
def input_map():
    beam_window = hp.gauss_beam(
        fwhm=(1 * u.deg).to_value(u.radian), lmax=LMAX
    ) ** 2
    cl = np.zeros((6, len(beam_window)))
    cl[0:3] = beam_window
    np.random.seed(7)
    return hp.synfast(cl, NSIDE, lmax=LMAX, new=True) * u.uK_RJ


def smooth_reference(map_in, fwhm):
    alm = map2alm(map_in, NSIDE, LMAX)
    hp.smoothalm(alm, fwhm=fwhm.to_value(u.rad), inplace=True, pol=True)
    return hp.alm2map(alm, nside=NSIDE, pixwin=False) * u.uK_RJ


def test_get_differential_fwhm():
    assert_quantity_allclose(
        get_differential_fwhm(TARGET_FWHM, PRE_FWHM),
        np.sqrt(TARGET_FWHM.to_value(u.rad) ** 2 - PRE_FWHM.to_value(u.rad) ** 2)
        * u.rad,
    )
    assert_quantity_allclose(
        get_differential_fwhm("1 deg", "56 arcmin"),
        get_differential_fwhm(1 * u.deg, 56 * u.arcmin),
    )
    assert get_differential_fwhm(PRE_FWHM, PRE_FWHM).value == 0
    assert get_differential_fwhm(30 * u.arcmin, PRE_FWHM).value == 0


def test_differential_smoothing_from_attribute(input_map):
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = PRE_FWHM
    smoothed = apply_smoothing_and_coord_transform(
        tagged, fwhm=TARGET_FWHM, lmax=LMAX
    )
    assert_quantity_allclose(
        smoothed,
        smooth_reference(
            input_map, get_differential_fwhm(TARGET_FWHM, PRE_FWHM)
        ),
    )
    assert_quantity_allclose(
        smoothed,
        apply_smoothing_and_coord_transform(
            input_map,
            fwhm=get_differential_fwhm(TARGET_FWHM, PRE_FWHM),
            lmax=LMAX,
        ),
    )
    assert smoothed.pre_applied_fwhm == TARGET_FWHM


def test_differential_smoothing_from_explicit_argument(input_map):
    smoothed = apply_smoothing_and_coord_transform(
        input_map, fwhm=TARGET_FWHM, lmax=LMAX, pre_applied_fwhm="56 arcmin"
    )
    assert_quantity_allclose(
        smoothed,
        smooth_reference(
            input_map, get_differential_fwhm(TARGET_FWHM, PRE_FWHM)
        ),
    )
    assert smoothed.pre_applied_fwhm == TARGET_FWHM


def test_explicit_argument_wins_over_attribute(input_map):
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = 30 * u.arcmin
    smoothed = apply_smoothing_and_coord_transform(
        tagged, fwhm=TARGET_FWHM, lmax=LMAX, pre_applied_fwhm=PRE_FWHM
    )
    assert_quantity_allclose(
        smoothed,
        smooth_reference(
            input_map, get_differential_fwhm(TARGET_FWHM, PRE_FWHM)
        ),
    )
    assert_quantity_allclose(
        apply_smoothing_and_coord_transform(
            tagged, fwhm=TARGET_FWHM, lmax=LMAX, pre_applied_fwhm=0 * u.deg
        ),
        smooth_reference(input_map, TARGET_FWHM),
    )


def test_differential_smoothing_equals_window_division(input_map):
    alm = map2alm(input_map, NSIDE, LMAX)
    beam = hp.gauss_beam(
        TARGET_FWHM.to_value(u.radian), lmax=LMAX, pol=True
    ) / hp.gauss_beam(PRE_FWHM.to_value(u.radian), lmax=LMAX, pol=True)
    for each_alm, each_beam in zip(alm, beam.T):
        hp.almxfl(each_alm, each_beam, inplace=True)
    reference = hp.alm2map(alm, nside=NSIDE, pixwin=False) * u.uK_RJ

    tagged = input_map.copy()
    tagged.pre_applied_fwhm = PRE_FWHM
    assert_quantity_allclose(
        apply_smoothing_and_coord_transform(
            tagged, fwhm=TARGET_FWHM, lmax=LMAX
        ),
        reference,
        rtol=1e-6,
    )


def test_resolution_floor_noop(input_map, caplog):
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = TARGET_FWHM
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        smoothed = apply_smoothing_and_coord_transform(
            tagged, fwhm=PRE_FWHM, lmax=LMAX
        )
    assert "cannot be deconvolved" in caplog.text
    assert_quantity_allclose(
        smoothed, apply_smoothing_and_coord_transform(tagged, lmax=LMAX)
    )
    assert smoothed.pre_applied_fwhm == TARGET_FWHM

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        apply_smoothing_and_coord_transform(
            tagged, fwhm=TARGET_FWHM, lmax=LMAX
        )
    assert "cannot be deconvolved" in caplog.text


def test_untagged_map_unchanged(input_map):
    smoothed = apply_smoothing_and_coord_transform(
        input_map, fwhm=TARGET_FWHM, lmax=LMAX
    )
    assert not hasattr(smoothed, "pre_applied_fwhm")
    assert_quantity_allclose(smoothed, smooth_reference(input_map, TARGET_FWHM))


def test_tag_preserved_without_smoothing(input_map):
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = PRE_FWHM
    output = apply_smoothing_and_coord_transform(tagged, lmax=LMAX)
    assert output.pre_applied_fwhm == PRE_FWHM


def test_chained_smoothing_stays_differential(input_map):
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = PRE_FWHM
    first = apply_smoothing_and_coord_transform(
        tagged, fwhm=TARGET_FWHM, lmax=LMAX
    )
    assert first.pre_applied_fwhm == TARGET_FWHM
    second = apply_smoothing_and_coord_transform(
        first, fwhm=LARGER_FWHM, lmax=LMAX
    )
    assert second.pre_applied_fwhm == LARGER_FWHM
    assert_quantity_allclose(
        second,
        smooth_reference(
            input_map, get_differential_fwhm(LARGER_FWHM, PRE_FWHM)
        ),
        rtol=1e-5,
    )


def test_beam_window_ignores_pre_applied_beam(input_map, caplog):
    beam_window = hp.gauss_beam(
        LARGER_FWHM.to_value(u.radian), lmax=LMAX, pol=True
    ).T
    tagged = input_map.copy()
    tagged.pre_applied_fwhm = PRE_FWHM
    with caplog.at_level(logging.WARNING, logger="pysm3"):
        smoothed = apply_smoothing_and_coord_transform(
            tagged, lmax=LMAX, beam_window=beam_window
        )
    assert "ignored" in caplog.text
    assert_quantity_allclose(
        smoothed,
        apply_smoothing_and_coord_transform(
            input_map, lmax=LMAX, beam_window=beam_window
        ),
    )


def test_invalid_pre_applied_fwhm(input_map):
    with pytest.raises(ValueError):
        apply_smoothing_and_coord_transform(
            input_map, fwhm=TARGET_FWHM, lmax=LMAX, pre_applied_fwhm="-1 deg"
        )
    with pytest.raises(UnitConversionError):
        apply_smoothing_and_coord_transform(
            input_map, fwhm=TARGET_FWHM, lmax=LMAX, pre_applied_fwhm="1 GHz"
        )


def make_powerlaw(pre_applied_fwhm=None, nside=32):
    np.random.seed(12)
    map_I = u.Quantity(
        np.random.rand(hp.nside2npix(nside)) * 100, u.uK_RJ
    )
    return pysm3.PowerLaw(
        map_I=map_I,
        freq_ref_I=23 * u.GHz,
        map_pl_index=-3.0,
        nside=nside,
        has_polarization=False,
        pre_applied_fwhm=pre_applied_fwhm,
    )


class ConstantModel(pysm3.Model):
    def get_emission(self, freqs, weights=None):
        return np.ones((3, hp.nside2npix(self.nside))) << u.uK_RJ


class TupleOutputModel(pysm3.Model):
    def get_emission(self, freqs, weights=None):
        return (np.ones((3, hp.nside2npix(self.nside))) << u.uK_RJ, "car")


def test_any_model_subclass_auto_tags_output():
    model = ConstantModel(nside=32, pre_applied_fwhm="56 arcmin")
    assert model.pre_applied_fwhm == 56 * u.arcmin
    assert model.includes_smoothing
    output = model.get_emission(30 * u.GHz)
    assert output.pre_applied_fwhm == 56 * u.arcmin


def test_any_model_subclass_default_no_tag():
    model = ConstantModel(nside=32)
    assert model.pre_applied_fwhm is None
    assert not model.includes_smoothing
    output = model.get_emission(30 * u.GHz)
    assert not hasattr(output, "pre_applied_fwhm")


def test_tuple_output_tagged_elementwise():
    output = TupleOutputModel(
        nside=32, pre_applied_fwhm="1 deg"
    ).get_emission(30 * u.GHz)
    assert output[0].pre_applied_fwhm == 1 * u.deg
    assert output[1] == "car"


def test_component_config_pre_applied_fwhm_any_model():
    np.random.seed(12)
    config = {
        "test_mbb": {
            "class": "ModifiedBlackBody",
            "map_I": u.Quantity(
                np.random.rand(3, hp.nside2npix(32)) * 100, u.uK_RJ
            ),
            "freq_ref_I": "545 GHz",
            "freq_ref_P": "545 GHz",
            "map_mbb_index": 1.54,
            "map_mbb_temperature": 20.0,
            "unit_mbb_temperature": "K",
            "pre_applied_fwhm": "56 arcmin",
        }
    }
    sky = pysm3.Sky(nside=32, component_config=config)
    assert sky.pre_applied_fwhm == 56 * u.arcmin
    output = sky.get_emission(100 * u.GHz)
    assert output.pre_applied_fwhm == 56 * u.arcmin
    assert "pre_applied_fwhm" in config["test_mbb"]


def test_sky_component_objects_any_model():
    sky = pysm3.Sky(
        nside=32,
        component_objects=[
            ConstantModel(nside=32, pre_applied_fwhm="1 deg"),
        ],
    )
    assert sky.pre_applied_fwhm == 1 * u.deg
    assert sky.get_emission(30 * u.GHz).pre_applied_fwhm == 1 * u.deg


def make_modified_black_body(pre_applied_fwhm=None):
    np.random.seed(12)
    return pysm3.ModifiedBlackBody(
        map_I=u.Quantity(
            np.random.rand(3, hp.nside2npix(32)) * 100, u.uK_RJ
        ),
        freq_ref_I=545 * u.GHz,
        freq_ref_P=545 * u.GHz,
        map_mbb_index=1.54,
        map_mbb_temperature=20.0,
        unit_mbb_temperature="K",
        nside=32,
        pre_applied_fwhm=pre_applied_fwhm,
    )


def test_direct_construction_any_model():
    model = make_modified_black_body("56 arcmin")
    assert model.pre_applied_fwhm == 56 * u.arcmin
    assert model.includes_smoothing
    output = model.get_emission(100 * u.GHz)
    assert output.pre_applied_fwhm == 56 * u.arcmin


def test_direct_construction_default_no_tag():
    model = make_modified_black_body()
    assert model.pre_applied_fwhm is None
    output = model.get_emission(100 * u.GHz)
    assert not hasattr(output, "pre_applied_fwhm")


def test_direct_construction_invalid_pre_applied_fwhm():
    with pytest.raises(UnitConversionError):
        make_modified_black_body("1 GHz")


def test_sky_rejects_pre_applied_fwhm():
    with pytest.raises(ValueError, match="derives pre_applied_fwhm"):
        pysm3.Sky(
            nside=32,
            component_objects=[make_powerlaw()],
            pre_applied_fwhm="1 deg",
        )


def test_cmb_dipole_supports_pre_applied_fwhm():
    dipole = pysm3.CMBDipole(
        nside=32,
        amp="3366.6 uK_CMB",
        T_cmb="2.725 K_CMB",
        dip_lon="263.986 deg",
        dip_lat="48.247 deg",
        pre_applied_fwhm="1 deg",
    )
    assert dipole.pre_applied_fwhm == 1 * u.deg
    assert dipole.includes_smoothing
    output = dipole.get_emission(70 * u.GHz)
    assert output.pre_applied_fwhm == 1 * u.deg


def test_interpolating_component_pre_applied_fwhm(tmp_path):
    nside = 32
    np.random.seed(12)
    for freq in [20.0, 30.0]:
        hp.write_map(
            str(tmp_path / f"{freq:05.1f}.fits"),
            np.random.rand(3, hp.nside2npix(nside)) * 100,
            overwrite=True,
        )
    interp = pysm3.InterpolatingComponent(
        path=str(tmp_path),
        input_units="uK_RJ",
        nside=nside,
        freqs=[20.0, 30.0],
        pre_applied_fwhm="56 arcmin",
    )
    assert interp.includes_smoothing
    output = interp.get_emission(25 * u.GHz)
    assert output.pre_applied_fwhm == 56 * u.arcmin


def test_powerlaw_tags_output():
    model = make_powerlaw("56 arcmin")
    assert model.pre_applied_fwhm == 56 * u.arcmin
    assert model.includes_smoothing
    output = model.get_emission(30 * u.GHz)
    assert output.shape == (3, hp.nside2npix(32))
    assert output.pre_applied_fwhm == 56 * u.arcmin


def test_powerlaw_default_has_no_tag():
    model = make_powerlaw()
    assert model.pre_applied_fwhm is None
    assert not model.includes_smoothing
    output = model.get_emission(30 * u.GHz)
    assert not hasattr(output, "pre_applied_fwhm")


def test_curved_power_law_tags_output():
    model = make_powerlaw("1 deg")
    curved = pysm3.CurvedPowerLaw(
        map_I=model.I_ref,
        freq_ref_I=23 * u.GHz,
        map_pl_index=-3.0,
        nside=32,
        spectral_curvature=-0.05,
        freq_curve="23 GHz",
        has_polarization=False,
        pre_applied_fwhm="1 deg",
    )
    output = curved.get_emission(30 * u.GHz)
    assert output.pre_applied_fwhm == 1 * u.deg


def test_invalid_model_pre_applied_fwhm():
    with pytest.raises(ValueError):
        make_powerlaw("-1 deg")
    with pytest.raises(UnitConversionError):
        make_powerlaw("1 GHz")


def test_sky_single_component(input_map):
    sky = pysm3.Sky(nside=32, component_objects=[make_powerlaw("1 deg")])
    assert sky.pre_applied_fwhm == 1 * u.deg
    assert sky.includes_smoothing
    output = sky.get_emission(30 * u.GHz)
    assert output.pre_applied_fwhm == 1 * u.deg


def test_sky_all_components_without_beam():
    sky = pysm3.Sky(
        nside=32,
        component_objects=[make_powerlaw(), make_powerlaw()],
    )
    assert sky.pre_applied_fwhm is None
    assert not sky.includes_smoothing
    output = sky.get_emission(30 * u.GHz)
    assert not hasattr(output, "pre_applied_fwhm")


def test_sky_same_value_components():
    sky = pysm3.Sky(
        nside=32,
        component_objects=[
            make_powerlaw("1 deg"),
            make_powerlaw(u.Quantity(60, u.arcmin)),
        ],
    )
    assert sky.pre_applied_fwhm == 1 * u.deg
    output = sky.get_emission(30 * u.GHz)
    assert output.pre_applied_fwhm == 1 * u.deg


def test_sky_mixed_components_raise():
    with pytest.raises(ValueError, match="same pre_applied_fwhm"):
        pysm3.Sky(
            nside=32,
            component_objects=[make_powerlaw("1 deg"), make_powerlaw()],
        )


def test_sky_add_component():
    sky = pysm3.Sky(nside=32, component_objects=[make_powerlaw("1 deg")])
    sky.add_component(make_powerlaw("1 deg"))
    assert len(sky.components) == 2
    with pytest.raises(ValueError, match="same pre_applied_fwhm"):
        sky.add_component(make_powerlaw())
    assert len(sky.components) == 2


def test_sky_nested():
    inner = pysm3.Sky(nside=32, component_objects=[make_powerlaw("1 deg")])
    sky = pysm3.Sky(
        nside=32, component_objects=[inner, make_powerlaw("1 deg")]
    )
    assert sky.pre_applied_fwhm == 1 * u.deg
    with pytest.raises(ValueError, match="same pre_applied_fwhm"):
        pysm3.Sky(nside=32, component_objects=[inner, make_powerlaw()])


def test_component_config_pre_applied_fwhm():
    np.random.seed(12)
    sky = pysm3.Sky(
        nside=32,
        component_config={
            "test_pl": {
                "class": "PowerLaw",
                "map_I": u.Quantity(
                    np.random.rand(hp.nside2npix(32)) * 100, u.uK_RJ
                ),
                "freq_ref_I": "23 GHz",
                "map_pl_index": -3.0,
                "has_polarization": False,
                "pre_applied_fwhm": "56 arcmin",
            }
        },
    )
    assert sky.pre_applied_fwhm == 56 * u.arcmin
    output = sky.get_emission(30 * u.GHz)
    assert output.pre_applied_fwhm == 56 * u.arcmin


def test_end_to_end_model_to_smoothing():
    sky = pysm3.Sky(nside=32, component_objects=[make_powerlaw("56 arcmin")])
    emission = sky.get_emission(30 * u.GHz)
    smoothed = apply_smoothing_and_coord_transform(
        emission, fwhm=1 * u.deg, lmax=48
    )
    plain = emission.value * u.uK_RJ
    assert_quantity_allclose(
        smoothed,
        apply_smoothing_and_coord_transform(
            plain, fwhm=get_differential_fwhm(1 * u.deg, 56 * u.arcmin), lmax=48
        ),
    )
    assert smoothed.pre_applied_fwhm == 1 * u.deg


def test_get_common_pre_applied_fwhm_empty():
    assert get_common_pre_applied_fwhm([]) is None
