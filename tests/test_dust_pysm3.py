import psutil
import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3
from pysm3 import units as u
from pysm3.models.dust import blackbody_ratio


@pytest.mark.parametrize("model_tag", ["d9", "d10"])
def test_dust_model_353(model_tag):
    nside = 2048

    freq = 353 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=nside)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"dust_gnilc/gnilc_dust_template_nside{nside}_2023.02.10.fits",
        nside=nside,
        field=(0, 1, 2),
    )
    rtol = 1e-5

    # if model_tag == "d11":
    #    beam = 1 * u.deg
    #    input_template = hp.smoothing(input_template, fwhm=beam.to_value(u.radians))
    #    output = hp.smoothing(output, fwhm=beam.to_value(u.radians))
    #    rtol = 1e-2

    assert_quantity_allclose(input_template, output, rtol=rtol)


@pytest.mark.parametrize("model_tag", ["d9", "d10"])
def test_gnilc_857(model_tag):
    freq = 857 * u.GHz

    model = pysm3.Sky(preset_strings=[model_tag], nside=2048)

    output = model.get_emission(freq)

    input_template = pysm3.models.read_map(
        f"dust_gnilc/gnilc_dust_template_nside{2048}_2023.02.10.fits",
        nside=2048,
        field=(0, 1, 2),
    )

    freq_ref = 353 * u.GHz
    beta = (
        1.48
        if model_tag == "d9"
        else pysm3.models.read_map(
            f"dust_gnilc/gnilc_dust_beta_nside{2048}_2023.06.06.fits",
            nside=2048,
            field=0,
        )
    )
    Td = (
        19.6 * u.K
        if model_tag == "d9"
        else pysm3.models.read_map(
            f"dust_gnilc/gnilc_dust_Td_nside{2048}_2023.06.06.fits",
            nside=2048,
            field=0,
        )
    )
    scaling = (freq / freq_ref) ** (beta - 2)
    scaling *= blackbody_ratio(freq.value, freq_ref.value, Td.to_value(u.K))

    assert_quantity_allclose(input_template * scaling, output, rtol=1e-6)


@pytest.mark.skipif(
    psutil.virtual_memory().total * u.byte < 20 * u.GB,
    reason="Running d11 at high lmax requires 20 GB of RAM",
)
@pytest.mark.parametrize("freq", [353, 857])
def test_d10_vs_d11(freq):
    nside = 2048
    freq = freq * u.GHz

    output_d10 = pysm3.Sky(preset_strings=["d10"], nside=nside).get_emission(freq)
    d11_configuration = pysm3.sky.PRESET_MODELS["d11"].copy()
    del d11_configuration["class"]
    d11 = pysm3.models.ModifiedBlackBodyRealization(
        nside=nside,
        seeds=[8192, 777, 888],
        synalm_lmax=int(8192 * 2),
        **d11_configuration,
    )
    output_d11 = d11.get_emission(freq)

    rtol = 1e-5

    assert_quantity_allclose(output_d10, output_d11, rtol=rtol, atol=0.05 * u.uK_RJ)
