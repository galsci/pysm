import sys
sys.path.append('/home/ben/Projects/PySM/PySM_public')
import unittest
import numpy as np
import pysm.component_models.galactic.dust as dust
import astropy.units as units
from astropy.units import UnitsError

class TestModifiedBlackBody(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_emission(self):
        return

class TestDecorrelatedModifiedBlackBody(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_emission(self):
        return

class TestDecorrelationMatrices(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

class TestInvertSafe(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        self.input = None
        return

    def test_inversion(self):
        dust.invert_safe(self.input)
        return

class TestBlackbodyRatio(unittest.TestCase):
    def setUp(self):
        self.nu_from = 100. * units.GHz
        self.nu_to = 400. * units.GHz
        self.temp = 20. * units.K

        self.nu_from_unitless = 100.
        self.nu_to_unitless = 400.
        self.temp_unitless = 20.

        self.nu_from_wrong_unit = 100. * units.K
        self.nu_to_wrong_unit = 400. * units.K
        self.temp_wrong_unit = 20. * units.s

    def test_blackbody_ratio(self):
        dust.blackbody_ratio(self.nu_to, self.nu_from, self.temp)
        dust.blackbody_ratio(self.nu_to, self.nu_from, self.temp)
        self.assertRaises(UnitsError, dust.blackbody_ratio, self.nu_to,
                          self.nu_from_wrong_unit, self.temp)
        self.assertRaises(TypeError, dust.blackbody_ratio, self.nu_to,
                          self.nu_from_unitless, self.temp)
        self.assertRaises(units.UnitsError, dust.blackbody_ratio,
                          self.nu_to_wrong_unit, self.nu_from, self.temp)
        self.assertRaises(TypeError, dust.blackbody_ratio,
                          self.nu_to_unitless, self.nu_from, self.temp)
        self.assertRaises(units.UnitsError, dust.blackbody_ratio, self.nu_to,
                          self.nu_from, self.temp_wrong_unit)
        self.assertRaises(TypeError, dust.blackbody_ratio, self.nu_to,
                          self.nu_from, self.temp_unitless)
        
if __name__ == '__main__':
    unittest.main()
