from pathlib import Path
import unittest
import numpy as np
import healpy as hp
import pysm.units as units
from pysm.models.template import read_map


class TestUnits(unittest.TestCase):
    def setUp(self):
        self.T_CMB = 100.0 * units.K_CMB
        self.T_RJ = 100.0 * units.K_RJ
        self.freqs = 100.0 * units.GHz
        self.freqs_arr = np.array([10.0, 100.0, 1000.0]) * units.GHz
        self.temp_dir = Path(__file__).absolute().parent / "temp_dir"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_fits_file_RJ = self.temp_dir / "test_RJ.fits"
        self.temp_fits_file_CMB = self.temp_dir / "test_CMB.fits"
        self.temp_fits_file_dimless = self.temp_dir / "test_dimless.fits"
        self.temp_fits_file_no_unit_hdr = self.temp_dir / "test_no_hdr.fits"
        nside = 256
        npix = hp.nside2npix(nside)
        self.test_map_RJ = np.random.randn(npix) * units.K_RJ
        self.test_map_CMB = np.random.randn(npix) * units.K_CMB
        self.test_map_dimless = units.Quantity(np.random.randn(npix), "")

    def tearDown(self):
        try:
            self.temp_fits_file_RJ.unlink()
        except FileNotFoundError:
            pass
        try:
            self.temp_fits_file_CMB.unlink()
        except FileNotFoundError:
            pass
        try:
            self.temp_fits_file_dimless.unlink()
        except FileNotFoundError:
            pass
        try:
            self.temp_fits_file_no_unit_hdr.unlink()
        except FileNotFoundError:
            pass
        self.temp_dir.rmdir()

    def test_conversion(self):
        """ Here we test that the numerical value of the conversion is correct.
        The mathematical form is

        ..math::
        I_\nu = \frac{2 \nu^2 k T_{\rm RJ}}{c^2} \\
        I_\nu = T_{\rm CMB} B^\prime_\nu(T_{\rm CMB, 0})

        so, eliminating the flux in this equation:

        ..math::
        T_{\rm RJ} / T_{\rm CMB} = \frac{c^2}{2 \nu^2 k_B}B^\prime_\nu(T_{\rm CMB, 0})

        Here we calculate the RHS of this equation and compare it to the
        ratio of T_RJ and the result of its transformation to T_CMB.
        """
        equiv = {"equivalencies": units.cmb_equivalencies(self.freqs)}
        rj_from_cmb = self.T_CMB.to(units.K_RJ, **equiv)
        cmb_from_rj = self.T_RJ.to(units.K_CMB, **equiv)

        # check that the reverse transformation gives overall transformation of unity.
        reverse1 = rj_from_cmb.to(units.K_CMB, **equiv)
        reverse2 = cmb_from_rj.to(units.K_RJ, **equiv)

        np.testing.assert_almost_equal(1.0, self.T_CMB / reverse1, decimal=6)
        np.testing.assert_almost_equal(1.0, self.T_RJ / reverse2, decimal=6)

    def test_fits_unit_funcitonality(self):
        """ Test that the units can be written to the fits header. Check that
        they can be read in again and assigned to the data in that fits file
        correctly.
        """
        hp.write_map(
            str(self.temp_fits_file_RJ),
            self.test_map_RJ.value,
            column_units=self.test_map_RJ.unit.to_string("generic"),
        )
        hp.write_map(
            str(self.temp_fits_file_CMB),
            self.test_map_CMB.value,
            column_units=self.test_map_CMB.unit.to_string("generic"),
        )
        hp.write_map(
            str(self.temp_fits_file_dimless),
            self.test_map_dimless.value,
            column_units=self.test_map_dimless.unit.to_string("generic"),
        )
        hp.write_map(str(self.temp_fits_file_no_unit_hdr), self.test_map_dimless.value)

        cmb_in = read_map(str(self.temp_fits_file_CMB), 256)
        rj_in = read_map(str(self.temp_fits_file_RJ), 256)
        dimless_in = read_map(str(self.temp_fits_file_dimless), 256)
        no_unit_hdr = read_map(str(self.temp_fits_file_no_unit_hdr), 256)
        self.assertTrue(cmb_in.unit == units.K_CMB)
        self.assertTrue(rj_in.unit == units.K_RJ)
        self.assertTrue(dimless_in.unit == units.dimensionless_unscaled)
        self.assertTrue(no_unit_hdr.unit == units.dimensionless_unscaled)
        return


if __name__ == "__main__":
    unittest.main()
