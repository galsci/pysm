import pysm3


def test_sky_max_nside():
    sky = pysm3.Sky(nside=32, preset_strings=["co1", "s1"])
    for component in sky.components:
        assert component.max_nside == 512


def test_sky_max_nside_highres():
    sky = pysm3.Sky(nside=32, preset_strings=["d10", "s5"])
    for component in sky.components:
        assert component.max_nside == 8192
