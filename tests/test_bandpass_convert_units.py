import unittest

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from scipy import constants

from pysm3 import bandpass_unit_conversion, utils
from pysm3 import units as u


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
        planck_HFI_file = utils.RemoteData().get(
            "pysm_2_test_data/HFI_RIMO_R1.10_onlybandpasses.fits.gz"
        )
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
            good = np.logical_and(frequency > 1, frequency < 1200)
            transmission = np.array(transmission)
            freqs, weights = frequency[good], transmission[good]
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

        # Comparison with PySM 2
        self.CMB2MJysr_avg_pysm2 = {}
        self.CMB2MJysr_avg_pysm2[100] = 243.1769177398688 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_pysm2[143] = 376.0354144258313 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_pysm2[217] = 476.8415133279352 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_pysm2[353] = 282.97356344925504 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_pysm2[545] = 57.27399314424877 * u.MJy / u.sr / u.K_CMB
        # 857 integration gives nan in PySM 2

        # Comparison with @keskitalo's `tod2flux` removing the nu^-1 IRAS convention
        self.CMB2MJysr_avg_tod2flux = {}
        self.CMB2MJysr_avg_tod2flux[100] = 243.08197 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_tod2flux[143] = 375.959 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_tod2flux[217] = 476.18509 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_tod2flux[353] = 283.31489 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_tod2flux[545] = 57.339909 * u.MJy / u.sr / u.K_CMB
        self.CMB2MJysr_avg_tod2flux[857] = 2.2449586 * u.MJy / u.sr / u.K_CMB

        """And for MJysr to K_RJ"""
        self.MJysr2KRJ_avg = {}
        self.MJysr2KRJ_avg[100] = 0.0032548074 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[143] = 0.0015916707 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[217] = 0.00069120334 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[353] = 0.00026120163 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[545] = 0.00010958025 * u.K_RJ / (u.MJy / u.sr)
        self.MJysr2KRJ_avg[857] = 4.4316316e-5 * u.K_RJ / (u.MJy / u.sr)

        # Comparison with @keskitalo's `tod2flux`
        self.MJysr2KRJ_avg_tod2flux = {}
        self.MJysr2KRJ_avg_tod2flux[100] = (
            0.0031315339458127182 * u.K_RJ / (u.MJy / u.sr)
        )
        self.MJysr2KRJ_avg_tod2flux[143] = (
            0.0015988463655762359 * u.K_RJ / (u.MJy / u.sr)
        )
        self.MJysr2KRJ_avg_tod2flux[217] = (
            0.0006436743956801712 * u.K_RJ / (u.MJy / u.sr)
        )
        self.MJysr2KRJ_avg_tod2flux[353] = (
            0.00024388323334674145 * u.K_RJ / (u.MJy / u.sr)
        )
        self.MJysr2KRJ_avg_tod2flux[545] = (
            0.00010246437528390336 * u.K_RJ / (u.MJy / u.sr)
        )
        self.MJysr2KRJ_avg_tod2flux[857] = (
            4.321471531508259e-05 * u.K_RJ / (u.MJy / u.sr)
        )

    def test_bandpass_unit_conversion_CMB2MJysr(self):
        """

        Also from `hfi_unit_conversion.pro` at
        https://irsa.ipac.caltech.edu/data/Planck/release_1/software/index.html
        It looks like Planck is averaging the coefficients, it is not computing
        the coefficient from the averaged bandpass as we are doing here.

        Moreover, the convention for the Planck HFI bandpass conversion is to
        use the IRAS convention of considering a spectrum of shape v^-1
        so the agreement to 5 or 6% looks reasonable.
        If we compare with Reijo Keskitalo's tod2flux, which implements the Planck
        algorithm and disable the nu^-1 factor in the integration, agreement is
        below 1% for all channels except 857GHz which is ~1.3%.
        """

        for freq in self.CMB2MJysr_avg:
            pysm_conv = bandpass_unit_conversion(
                self.channels[freq][0] * u.GHz,
                weights=self.channels[freq][1],
                input_unit=u.K_CMB,
                output_unit=u.MJy / u.sr,
            )
            assert_quantity_allclose(
                pysm_conv, self.CMB2MJysr_avg[freq], rtol=5 * u.pct
            )
            assert_quantity_allclose(
                pysm_conv,
                self.CMB2MJysr_avg_pysm2.get(freq, pysm_conv),
                rtol=0.02 * u.pct,
            )
            assert_quantity_allclose(
                pysm_conv,
                self.CMB2MJysr_avg_tod2flux.get(freq),
                rtol=1.3 * u.pct,
            )

    def test_bandpass_unit_conversion_MJysr2KRJ(self):

        for freq in self.MJysr2KRJ_avg:
            pysm_conv = bandpass_unit_conversion(
                self.channels[freq][0] * u.GHz,
                weights=self.channels[freq][1],
                input_unit=u.MJy / u.sr,
                output_unit=u.K_RJ,
            )
            assert_quantity_allclose(
                pysm_conv, self.MJysr2KRJ_avg[freq], rtol=6 * u.pct
            )
            assert_quantity_allclose(
                pysm_conv, self.MJysr2KRJ_avg_tod2flux[freq], rtol=2 * u.pct
            )


class test_bandpass_convert_integration(unittest.TestCase):
    def setUp(self):
        nsamples = 50
        nu2 = 40.0
        nu1 = 20.0

        self.freqs = np.linspace(nu1, nu2, nsamples) * u.GHz

        weights = np.ones(len(self.freqs))
        self.Jysr2CMB = trapezoid(weights, x=self.freqs) / trapezoid(
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
