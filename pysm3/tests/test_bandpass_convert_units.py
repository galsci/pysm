import unittest
import numpy as np
from astropy.io import fits
import scipy.constants as constants
from pysm3 import units as u
from pysm3 import bandpass_unit_conversion
from astropy.tests.helper import assert_quantity_allclose
from pysm3 import utils


class test_Bandpass_Unit_Conversion(unittest.TestCase):
    def setUp(self):
        """To test the bandpass unit conversion we use the Planck detector
        averaged bandpasses provided here:
        https://wiki.cosmos.esa.int/planckpla/index.php/The_RIMO. We
        compute the unit conversion factors for these bandpasses and
        compare them to the official Planck factors provided here:
        https://wiki.cosmos.esa.int/planckpla/index.php/UC_CC_Tables
        """
        # Read in the fits file. This contains only the HFI frequencies 100 -> 857.
        planck_HFI_file = utils.RemoteData().get("pysm_2_test_data/HFI_RIMO_R1.10_onlybandpasses.fits.gz")
        with fits.open(planck_HFI_file) as hdu:
            bandpasses = [np.array(hdu[i].data) for i in range(2, 8)]
            names = [int(hdu[i].name[-3:]) for i in range(2, 8)]
        # The table contains 4 lists: wavenumber, transmission, 1-sigma uncertainty, flag.
        # We are only interested in wavenumber and transmission.
        self.channels = {}
        for b, name in zip(bandpasses, names):
            wavenumber, transmission, _, _ = list(zip(*b))
            frequency = 1e-7 * constants.c * np.array(wavenumber)
            # exclude the element frqeuency[0] = 0
            filt = lambda x: (x[0] > 1.0) & (x[0] < 1200)
            freqs, weights = list(zip(*filter(filt, zip(frequency, transmission))))
            self.channels[name] = (np.array(freqs, dtype=np.double), np.array(weights))

        """Planck-provided coefficients for K_CMB to MJysr.
       These should only be taken to the first decimal place.

       """
        self.CMB2MJysr_avg = {}
        self.CMB2MJysr_avg[100] = 244.0960 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg[143] = 371.7327 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg[217] = 483.6874 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg[353] = 287.4517 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg[545] = 58.0356 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg[857] = 2.2681 * u.MJy / u.sr / u.K_CMB

        """And for MJysr to K_RJ"""
        self.MJysr2KRJ_avg = {}
        self.MJysr2KRJ_avg[100] = 0.0032548074 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[143] = 0.0015916707 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[217] = 0.00069120334 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[353] = 0.00026120163 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[545] = 0.00010958025 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[857] = 4.4316316e-5 * u.K_RJ / (u.MJy / u.sr)

    def test_bandpass_unit_conversion_CMB2MJysr(self):
        """Note that the precision is limited by uncertainty on the bandpass central frequency.

        Also from `hfi_unit_conversion.pro` at
        https://irsa.ipac.caltech.edu/data/Planck/release_1/software/index.html
        It looks like Planck is averaging the coefficients, it is not computing
        the coefficient from the averaged bandpass as we are doing here.

        So the agreement to 5 or 6% looks reasonable
        """

        for freq in self.CMB2MJysr_avg.keys():
            pysm_conv = bandpass_unit_conversion(
                self.channels[freq][0] * u.GHz,
                weights=self.channels[freq][1],
                input_unit=u.K_CMB,
                output_unit=u.MJy / u.sr,
            )
            assert_quantity_allclose(pysm_conv, self.CMB2MJysr_avg[freq], rtol=5 / 100)

    def test_bandpass_unit_conversion_MJysr2KRJ(self):

        for freq in self.MJysr2KRJ_avg.keys():
            pysm_conv = bandpass_unit_conversion(
                self.channels[freq][0] * u.GHz,
                weights=self.channels[freq][1],
                input_unit=u.MJy / u.sr,
                output_unit=u.K_RJ,
            )
            assert_quantity_allclose(pysm_conv, self.MJysr2KRJ_avg[freq], rtol=6 / 100)


class test_bandpass_convert_integration(unittest.TestCase):
    def setUp(self):
        nsamples = 50
        nu2 = 40.0
        nu1 = 20.0

        self.freqs = np.linspace(nu1, nu2, nsamples) * u.GHz

        weights = np.ones(len(self.freqs))
        self.Jysr2CMB = np.trapz(weights, x=self.freqs) / np.trapz(
            (1 * u.K_CMB).to(
                u.Jy / u.sr, equivalencies=u.cmb_equivalencies(self.freqs)
            ),
            x=self.freqs,
        )

    def test_bandpass_convert_units(self):
        Uc1 = bandpass_unit_conversion(
            output_unit=u.K_CMB, freqs=self.freqs, input_unit=u.Jy / u.sr
        )
        np.testing.assert_almost_equal(Uc1.value, self.Jysr2CMB.value)
