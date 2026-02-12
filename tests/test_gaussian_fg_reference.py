import numpy as np
import pytest

import pysm3
import pysm3.units as u
from pysm3.models.dust import blackbody_ratio


def _kcmb_to_rj(freq_ghz):
    with u.set_enabled_equivalencies(u.cmb_equivalencies(freq_ghz * u.GHz)):
        return (1.0 * u.uK_CMB).to_value(u.uK_RJ)


def _reference_dell(ells, amplitude, alpha):
    dell = np.zeros_like(ells, dtype=np.float64)
    valid = ells >= 2
    dell[valid] = amplitude * (ells[valid] / 80.0) ** alpha
    return dell


@pytest.mark.parametrize(
    ("component", "params", "freq_ghz"),
    [
        (
            pysm3.GaussianDust,
            {
                "amplitude_tt": 5600.0,
                "amplitude_ee": 56.0,
                "amplitude_bb": 28.0,
                "alpha_tt": -0.8,
                "alpha_ee": -0.32,
                "alpha_bb": -0.16,
                "freq_pivot": 353.0,
                "beta": 1.54,
                "temperature": 20.0,
            },
            145.0,
        ),
        (
            pysm3.GaussianSynchrotron,
            {
                "amplitude_tt": 100.0,
                "amplitude_ee": 9.0,
                "amplitude_bb": 1.6,
                "alpha_tt": -0.8,
                "alpha_ee": -0.7,
                "alpha_bb": -0.93,
                "freq_pivot": 23.0,
                "beta": -3.0,
            },
            30.0,
        ),
    ],
)
def test_dell_reference_agreement(component, params, freq_ghz):
    model = component(nside=32, seed=0, lmax=128)
    ells = np.arange(2, 129)
    theory = model.get_dell_theory(freq_ghz, ells=ells)

    if component is pysm3.GaussianDust:
        scaling = blackbody_ratio(freq_ghz, params["freq_pivot"], params["temperature"])
        scaling *= (freq_ghz / params["freq_pivot"]) ** (params["beta"] - 2.0)
        with u.set_enabled_equivalencies(u.cmb_equivalencies(params["freq_pivot"] * u.GHz)):
            pivot_kcmb_to_jysr = (1.0 * u.K_CMB).to_value(u.Jy / u.sr)
        with u.set_enabled_equivalencies(u.cmb_equivalencies(freq_ghz * u.GHz)):
            freq_kcmb_to_jysr = (1.0 * u.K_CMB).to_value(u.Jy / u.sr)
        scaling *= pivot_kcmb_to_jysr / freq_kcmb_to_jysr
    else:
        scaling = (freq_ghz / params["freq_pivot"]) ** params["beta"]
        with u.set_enabled_equivalencies(u.cmb_equivalencies(params["freq_pivot"] * u.GHz)):
            pivot_kcmb_to_jysr = (1.0 * u.K_CMB).to_value(u.Jy / u.sr)
        with u.set_enabled_equivalencies(u.cmb_equivalencies(freq_ghz * u.GHz)):
            freq_kcmb_to_jysr = (1.0 * u.K_CMB).to_value(u.Jy / u.sr)
        scaling *= pivot_kcmb_to_jysr / freq_kcmb_to_jysr

    amp_scale = (scaling * _kcmb_to_rj(freq_ghz)) ** 2

    expected_tt = _reference_dell(ells, params["amplitude_tt"], params["alpha_tt"]) * amp_scale
    expected_ee = _reference_dell(ells, params["amplitude_ee"], params["alpha_ee"]) * amp_scale
    expected_bb = _reference_dell(ells, params["amplitude_bb"], params["alpha_bb"]) * amp_scale

    np.testing.assert_allclose(theory["TT"], expected_tt)
    np.testing.assert_allclose(theory["EE"], expected_ee)
    np.testing.assert_allclose(theory["BB"], expected_bb)
