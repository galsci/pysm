import numpy as np
import pytest

import pysm3
import pysm3.units as u


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        (
            pysm3.GaussianDust,
            {
                "amplitude_ee": 56.0,
                "amplitude_bb": 28.0,
                "amplitude_tt": 5600.0,
                "alpha_ee": -0.32,
                "alpha_bb": -0.16,
                "alpha_tt": -0.8,
                "freq_pivot": 353.0,
            },
        ),
        (
            pysm3.GaussianSynchrotron,
            {
                "amplitude_ee": 9.0,
                "amplitude_bb": 1.6,
                "amplitude_tt": 100.0,
                "alpha_ee": -0.7,
                "alpha_bb": -0.93,
                "alpha_tt": -0.8,
                "freq_pivot": 23.0,
            },
        ),
    ],
)
def test_default_parameters(component, expected):
    model = component(nside=32, seed=3)
    assert model.amplitude_ee == expected["amplitude_ee"]
    assert model.amplitude_bb == expected["amplitude_bb"]
    assert model.amplitude_tt == expected["amplitude_tt"]
    assert model.alpha_ee == expected["alpha_ee"]
    assert model.alpha_bb == expected["alpha_bb"]
    assert model.alpha_tt == expected["alpha_tt"]
    assert np.isclose(model.freq_pivot.value, expected["freq_pivot"])


@pytest.mark.parametrize("component", [pysm3.GaussianDust, pysm3.GaussianSynchrotron])
def test_reproducible_with_seed(component):
    model_1 = component(nside=32, seed=7)
    model_2 = component(nside=32, seed=7)
    model_3 = component(nside=32, seed=8)

    np.testing.assert_allclose(model_1.Q_ref, model_2.Q_ref)
    np.testing.assert_allclose(model_1.U_ref, model_2.U_ref)

    assert not np.allclose(model_1.Q_ref, model_3.Q_ref)


@pytest.mark.parametrize(
    ("component", "expected_tt", "expected_ee", "expected_bb"),
    [
        (pysm3.GaussianDust, 5600.0, 56.0, 28.0),
        (pysm3.GaussianSynchrotron, 100.0, 9.0, 1.6),
    ],
)
def test_dell_normalization_at_ell_80(component, expected_tt, expected_ee, expected_bb):
    model = component(nside=32, lmax=128, seed=1)

    ell = 80
    dell_tt = ell * (ell + 1) / (2 * np.pi) * model.cl_tt[ell]
    dell_ee = ell * (ell + 1) / (2 * np.pi) * model.cl_ee[ell]
    dell_bb = ell * (ell + 1) / (2 * np.pi) * model.cl_bb[ell]

    assert np.isclose(dell_tt, expected_tt)
    assert np.isclose(dell_ee, expected_ee)
    assert np.isclose(dell_bb, expected_bb)


@pytest.mark.parametrize("preset", ["gd0", "gs0"])
def test_sky_preset_instantiation(preset):
    sky = pysm3.Sky(nside=16, preset_strings=[preset])
    emission = sky.get_emission(95 * u.GHz)

    assert emission.unit == u.uK_RJ
    assert emission.shape == (3, 12 * 16 * 16)
    assert np.nanstd(emission[0].value) > 0


@pytest.mark.parametrize("preset", ["gd0", "gs0"])
def test_sky_preset_deterministic(preset):
    sky_1 = pysm3.Sky(nside=16, preset_strings=[preset])
    sky_2 = pysm3.Sky(nside=16, preset_strings=[preset])
    emission_1 = sky_1.get_emission(95 * u.GHz)
    emission_2 = sky_2.get_emission(95 * u.GHz)

    np.testing.assert_allclose(emission_1.value, emission_2.value)


@pytest.mark.parametrize("component", [pysm3.GaussianDust, pysm3.GaussianSynchrotron])
def test_get_emission_bandpass(component):
    model = component(nside=16, seed=11)
    emission = model.get_emission(np.array([90.0, 100.0]) * u.GHz)

    assert emission.unit == u.uK_RJ
    assert emission.shape == (3, 12 * 16 * 16)
    assert np.nanstd(emission[0].value) > 0


def test_map_distribution_not_supported():
    with pytest.raises(NotImplementedError):
        pysm3.GaussianDust(nside=8, map_dist=object())
