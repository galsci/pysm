import unittest

from pysm3 import units as u


class TestConvertUnits(unittest.TestCase):

    def test_convert_units(self):
        a1 = 1 * u.uK_CMB.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(300.0 * u.GHz))
        a2 = 1 * u.uK_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(300.0 * u.GHz))
        self.assertAlmostEqual(1.0, a1 * a2)
        a1 = 1 * u.uK_CMB.to(
            u.Unit("MJy/sr"), equivalencies=u.cmb_equivalencies(300.0 * u.GHz)
        )
        a2 = 1 * u.Unit("MJy/sr").to(
            u.uK_CMB, equivalencies=u.cmb_equivalencies(300.0 * u.GHz)
        )
        self.assertAlmostEqual(1.0, a1 * a2)

        """Validation against ECRSC tables.
        https://irsasupport.ipac.caltech.edu/index.php?/Knowledgebase/
        Article/View/181/20/what-are-the-intensity-units-of-the-planck
        -all-sky-maps-and-how-do-i-convert-between-them
        These tables are based on the following tables:
        h = 6.626176e-26 erg*s
        k = 1.380662e-16 erg/L
        c = 2.997792458e1- cm/s
        T_CMB = 2.726
        The impact of the incorrect CMB temperature is especially impactful
        and limits some comparison to only ~2/3 s.f.
        """
        freqs = [30, 143, 857]
        uK_CMB_2_K_RJ = {}
        uK_CMB_2_K_RJ[30] = 9.77074e-7
        uK_CMB_2_K_RJ[143] = 6.04833e-7
        uK_CMB_2_K_RJ[857] = 6.37740e-11

        for freq in freqs:
            self.assertAlmostEqual(
                uK_CMB_2_K_RJ[freq],
                1
                * u.uK_CMB.to(u.K_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz)),
            )

        K_CMB_2_MJysr = {}
        K_CMB_2_MJysr[30] = 27.6515
        K_CMB_2_MJysr[143] = 628.272
        K_CMB_2_MJysr[857] = 22565.1

        for freq in freqs:
            self.assertAlmostEqual(
                K_CMB_2_MJysr[freq]
                / (
                    1
                    * u.K_RJ.to(
                        u.MJy / u.sr, equivalencies=u.cmb_equivalencies(freq * u.GHz)
                    )
                ),
                1.0,
                places=4,
            )

        # Note that the MJysr definition seems to match comparatively poorly. The
        # definitions of h, k, c in the document linked above are in cgs and differ
        # from those on wikipedia. This may conflict with the scipy constants I use.

        uK_CMB_2_MJysr = {}
        uK_CMB_2_MJysr[30] = 2.7e-5
        uK_CMB_2_MJysr[143] = 0.0003800
        uK_CMB_2_MJysr[857] = 1.43907e-6

        for freq in freqs:
            self.assertAlmostEqual(
                uK_CMB_2_MJysr[freq]
                / (
                    1
                    * u.uK_CMB.to(
                        u.MJy / u.sr, equivalencies=u.cmb_equivalencies(freq * u.GHz)
                    )
                ),
                1.0,
                places=2,
            )
