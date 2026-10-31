import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

import pysm3

NSIDE = 32
SEEDS = [8192, 777, 888]


@pytest.fixture(scope="module")
def d11():
    configuration = pysm3.sky.PRESET_MODELS["d11"].copy()
    del configuration["class"]
    return pysm3.models.ModifiedBlackBodyRealization(nside=NSIDE, seeds=SEEDS, **configuration)


def test_d13_preset_synalm_lmin():
    assert pysm3.sky.PRESET_MODELS["d13"]["synalm_lmin"] == 30


@pytest.mark.parametrize("synalm_lmin", [None, 0])
def test_synalm_lmin_no_cut_matches_d11(d11, synalm_lmin):
    # no lower cutoff must reproduce the original d11 realization exactly
    default = d11.draw_realization(seeds=SEEDS)
    cut = d11.draw_realization(synalm_lmin=synalm_lmin, seeds=SEEDS)
    for a, b in zip(default, cut):
        assert_quantity_allclose(a, b, rtol=0, atol=0 * a.unit)


def test_synalm_lmin_removes_stochastic_component(d11):
    # with every small-scale mode cut, the output is the large-scale template alone,
    # so it no longer depends on the seeds
    lmin = 3 * NSIDE
    out_a = d11.draw_realization(synalm_lmin=lmin, seeds=SEEDS)
    out_b = d11.draw_realization(synalm_lmin=lmin, seeds=[1, 2, 3])
    for a, b in zip(out_a, out_b):
        assert_quantity_allclose(a, b, rtol=1e-12)
    # while the uncut realization does depend on them
    full_b = d11.draw_realization(seeds=[1, 2, 3])
    assert not np.allclose(out_a[1].value, full_b[1].value)
