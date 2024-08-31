import healpy as hp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .. import units as u
from .. import utils
from .template import Model


class HensleyDraine2017(Model):
    """This is a model for modified black body emission.

    Attributes
    ----------
    I_ref, Q_ref, U_ref: ndarray
        Arrays containing the intensity or polarization reference
        templates at frequency `freq_ref_I` or `freq_ref_P`.
    """

    def __init__(
        self,
        map_I,
        map_Q,
        map_U,
        freq_ref_I,
        freq_ref_P,
        nside,
        max_nside=None,
        has_polarization=True,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        map_dist=None,
        mpi_comm=None,
        f_fe=None,
        f_car=None,
        rnd_uval=True,
        uval=0.2,
        nside_uval=256,
        seed=None,
    ):
        """This function initializes the Hensley-Draine 2017 model.

        The initialization of this model consists of:

        i) reading in emission templates from file, reading in data
        tables for the emissivity of silicate grains, silsicate grains
        with iron inclusions, and carbonaceous grains,

        ii) interpolating these tables across wavelength and
        interstellar radiation field (ISRF) strength,

        iii) generating a random realization of the interstellar
        radiation field, based on the modified Stefan-Boltzmann law,
        and measurements of dust temperature from Planck.

        Parameters
        ----------
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        unit_* : string or Unit
            Unit string or Unit object for all input FITS maps, if None,
            the input file should have a unit defined in the FITS header.
        freq_ref_I, freq_ref_P: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        nside: int
            Resolution parameter at which this model is to be calculated.
        f_fe: float
            Fractional composition of grain population with iron inclusions.
        f_car: float
            Fractional composition of grain population in carbonaceous grains.
        rnd_uval: bool (optional, default=True)
            Decide whether to draw a random realization of the ISRF.
        uval: float
            This value is used only if rnd_uval is False, the default of 0.2
            corresponds reasonably well to a Modifield Black Body model with
            temperature of 20K and an index of 1.54
        nside_uval: int (optional, default=256)
            HEALPix nside at which to evaluate the ISRF before ud_grade is applied
            to get the output scaling law. The default is the resolution at which
            the inputs available (COMMANDER dust beta and temperature).
        seed: int
            Number used to seed RNG for `uval`.
        """
        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        # do model setup
        self.I_ref = self.read_map(map_I, unit=unit_I)
        # This does unit conversion in place so we do not copy the data
        # we do not keep the original unit because otherwise we would need
        # to make a copy of the array when we run the model
        self.I_ref <<= u.uK_RJ
        self.freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
        self.has_polarization = has_polarization
        if has_polarization:
            self.Q_ref = self.read_map(map_Q, unit=unit_Q)
            self.Q_ref <<= u.uK_RJ
            self.U_ref = self.read_map(map_U, unit=unit_U)
            self.U_ref <<= u.uK_RJ
            self.freq_ref_P = u.Quantity(freq_ref_P).to(u.GHz)
        self.nside = int(nside)
        self.nside_uval = nside_uval

        self.f_fe = f_fe
        self.f_car = f_car
        self.f_sil = 1.0 - f_fe

        # break frequency below which model is not valid.
        self.__freq_break = 10.0 * u.GHz

        # data_sil contains the emission properties for silicon grains
        # with no iron inclusions.
        sil_data = self.read_txt("pysm_2/sil_fe00_2.0.dat")
        # data_silfe containts the emission properties for sillicon
        # grains with 5% iron inclusions.
        silfe_data = self.read_txt("pysm_2/sil_fe05_2.0.dat")
        # data_car contains the emission properties of carbonaceous
        # grains.
        car_data = self.read_txt("pysm_2/car_1.0.dat")

        # get the wavelengt (in microns) and dimensionless field
        # strengths over which these values were calculated.
        wav = sil_data[:, 0] * u.um

        uvec = np.arange(-3.0, 5.01, 0.1) * u.dimensionless_unscaled

        # The tabulated data is nu * I_nu / N_H, where N_H is the
        # number of hydrogen atoms per cm^2. Therefore the units of
        # the tabulated data are erg / s / sr.
        sil_data_i = (
            sil_data[:, 3:84]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)
        silfe_data_i = (
            silfe_data[:, 3:84]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)
        car_data_i = (
            car_data[:, 3:84]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)

        sil_data_p = (
            sil_data[:, 84:165]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)
        silfe_data_p = (
            silfe_data[:, 84:165]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)
        car_data_p = (
            car_data[:, 84:165]
            * u.erg
            / u.s
            / u.sr
            / wav[:, None].to(u.Hz, equivalencies=u.spectral())
        ).to(u.Jy / u.sr * u.cm**2)

        # interpolate the pre-computed solutions for the emissivity as a
        # function of grain 4 composition F_fe, Fcar, and field strenth U,
        # to get emissivity as a function of (U, wavelength).
        # Note that the spline, when evaluated, returns a unitless numpy array.
        # We will later ignore the cm^2 in the unit, since this does not affect
        # the outcome, and prevents the conversion between uK_RJ and Jy / sr
        # in astropy.
        assert sil_data_i.unit == u.Jy / u.sr * u.cm**2
        self.sil_i = RectBivariateSpline(uvec, wav, sil_data_i.T)

        assert silfe_data_i.unit == u.Jy / u.sr * u.cm**2
        self.car_i = RectBivariateSpline(uvec, wav, car_data_i.T)

        assert silfe_data_i.unit == u.Jy / u.sr * u.cm**2
        self.silfe_i = RectBivariateSpline(uvec, wav, silfe_data_i.T)

        assert sil_data_p.unit == u.Jy / u.sr * u.cm**2
        self.sil_p = RectBivariateSpline(uvec, wav, sil_data_p.T)

        assert car_data_p.unit == u.Jy / u.sr * u.cm**2
        self.car_p = RectBivariateSpline(uvec, wav, car_data_p.T)

        assert silfe_data_p.unit == u.Jy / u.sr * u.cm**2
        self.silfe_p = RectBivariateSpline(uvec, wav, silfe_data_p.T)

        # now draw the random realisation of uval if draw_uval = true
        if rnd_uval:
            T_mean = self.read_map(
                "pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits",
                unit="K",
                field=3,
                nside=self.nside_uval,
            )
            T_std = self.read_map(
                "pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits",
                unit="K",
                field=5,
                nside=self.nside_uval,
            )
            beta_mean = self.read_map(
                "pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits",
                unit="",
                field=6,
                nside=self.nside_uval,
            )
            beta_std = self.read_map(
                "pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits",
                unit="",
                field=8,
                nside=self.nside_uval,
            )
            # draw the realisations
            np.random.seed(seed)
            T = T_mean + np.random.randn(len(T_mean)) * T_std
            beta = beta_mean + np.random.randn(len(beta_mean)) * beta_std
            # use modified stefan boltzmann law to relate radiation field
            # strength to temperature and spectral index. Since the
            # interpolated data is only valid for -3 < uval <5 we clip the
            # generated values (the generated values are no where near these
            # limits, but it is good to note this for the future). We then
            # udgrade the uval map to whatever nside is being considered.
            # Since nside is not a parameter Sky knows about we have to get
            # it from A_I, which is not ideal.
            self.uval = hp.ud_grade(
                np.clip(
                    (4.0 + beta.value) * np.log10(T.value / np.mean(T.value)), -3.0, 5.0
                ),
                nside_out=nside,
            )
        else:
            self.uval = uval

        # compute the SED at the reference frequencies of the input templates.
        lambda_ref_i = self.freq_ref_I.to(u.um, equivalencies=u.spectral())
        lambda_ref_p = self.freq_ref_P.to(u.um, equivalencies=u.spectral())
        self.i_sed_at_nu0 = (
            (
                self.f_sil * self.sil_i.ev(self.uval, lambda_ref_i)
                + self.f_car * self.car_i.ev(self.uval, lambda_ref_i)
                + self.f_fe * self.silfe_i.ev(self.uval, lambda_ref_i)
            )
            * u.Jy
            / u.sr
        )

        self.p_sed_at_nu0 = (
            (
                self.f_sil * self.sil_p.ev(self.uval, lambda_ref_p)
                + self.f_car * self.car_p.ev(self.uval, lambda_ref_p)
                + self.f_fe * self.silfe_p.ev(self.uval, lambda_ref_p)
            )
            * u.Jy
            / u.sr
        )

    @u.quantity_input
    def evaluate_hd17_model_scaling(self, freq: u.GHz):
        """Method to evaluate the frequency scaling in the HD17 model. This
        caluculates the scaling factor to be applied to a set of T, Q, U maps
        in uK_RJ at some reference frequencies `self.freq_ref_I`,
        `self.freq_ref_P`, in order to scale them to frequencies `freqs`.

        Parameters
        ----------
        freq: float
            Frequency, convertible to microns, at which scaling factor is to
            be calculated.

        Returns
        -------
        tuple(ndarray)
            Scaling factor for intensity and polarization, at frequency
            `freq`. Tuple contains two arrays, each with shape (number of pixels).
        """
        freq = utils.check_freq_input(freq) * u.GHz
        # interpolation over pre-computed model is done in microns, so first convert
        # to microns.
        wav = freq.to(u.um, equivalencies=u.spectral())
        # evaluate the SED, which is currently does the scaling assuming Jy/sr.
        # uval is unitless, and lambdas are un microns.
        scaling_i = (
            self.f_sil * self.sil_i.ev(self.uval, wav)
            + self.f_car * self.car_i.ev(self.uval, wav)
            + self.f_fe * self.silfe_i.ev(self.uval, wav)
        ) / self.i_sed_at_nu0
        scaling_p = (
            self.f_sil * self.sil_p.ev(self.uval, wav)
            + self.f_car * self.car_p.ev(self.uval, wav)
            + self.f_fe * self.silfe_p.ev(self.uval, wav)
        ) / self.p_sed_at_nu0
        # scaling_i, and scaling_p are unitless scaling factors. However the scaling
        # does have the assumption of Jy / sr in the output map. We now account for
        # this by multiplying by the ratio of unit conversions from Jy / sr to uK_RJ
        # at the observed frequencies compared to the reference frequencies in
        # temperature and polarization.
        scaling_i *= (u.Jy / u.sr).to(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(freq)
        ) / (u.Jy / u.sr).to(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(self.freq_ref_I)
        )
        scaling_p *= (u.Jy / u.sr).to(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(freq)
        ) / (u.Jy / u.sr).to(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(self.freq_ref_P)
        )
        return scaling_i.value, scaling_p.value

    @u.quantity_input
    def evaluate_mbb_scaling(self, freq: u.GHz):
        """Method to evaluate a simple MBB scaling model with a constant
        index of 1.54. This method is used for frequencies below the break
        frequency (nominally 10 GHz), as the data the HD17 model relies upon
        stops at 10 GHz.

        At these frequencies, dust emission is largely irrelevant compared to
        other low frequency foregrounds, and so we do not expect the modeling
        assumptions to be significant. We therefore use a Rayleigh Jeans model
        for simplicity, and fix scale it from the SED at the break frequency.

        Parameters
        ----------
        freq: float
            Frequency at which to evaluate model (convertible to GHz).

        Returns
        -------
        tuple(ndarray)
            Scaling factor for intensity and polarization, at frequency
            `freq`. Tuple contains two arrays, each with shape (number of pixels).
        """
        # At these frequencies dust is largely irrelevant, and so we just
        # use a Rayleigh-Jeans model with constant spectral index of 1.54
        # for simplicity.
        RJ_factor = (freq / self.__freq_break) ** 1.54
        # calculate the HD17  model at the break frequency.
        scaling_i_at_cutoff, scaling_p_at_cutoff = self.evaluate_hd17_model_scaling(
            self.__freq_break
        )
        return (
            scaling_i_at_cutoff * RJ_factor.value,
            scaling_p_at_cutoff * RJ_factor.value,
        )

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
        """This function calculates the model of Hensley and Draine 2017 for
        the emission of a mixture of silicate, cabonaceous, and silicate
        grains with iron inclusions.

        Parameters
        ----------
        freqs: float
            Frequencies in GHz. When an array is passed, this is treated
            as a specification of a bandpass, and the bandpass average is
            calculated. For a single frequency, the emission at that
            frequency is returned (delta bandpass assumption).

        Returns
        -------
        ndarray
            Maps of T, Q, U for the given frequency specification.

        Notes
        -----
        If `weights` is not given, a flat bandpass is assumed. If `weights`
        is specified, it is automatically normalized.
        """
        freqs = utils.check_freq_input(freqs) * u.GHz
        # if `weights` is None, then this evenly weights all frequencies.
        weights = utils.normalize_weights(freqs, weights)
        output = np.zeros((3, len(self.I_ref)), dtype=self.I_ref.dtype)
        if len(freqs) > 1:
            # when `freqs` is an array, this is treated as an specification
            # of a bandpass. Definte `temp` to be an array in which the
            # average over the bandpass is accumulated.
            temp = np.zeros((3, len(self.I_ref)), dtype=self.I_ref.dtype)
        else:
            # when a single frequency is requested, `output` is just the
            # result of a single iteration of the loop below, so `temp`
            # and `output` are the same.
            temp = output
        # loop over frequencies. In each iteration evaluate the emission
        # in T, Q, U, at that frequency, and accumulate it in `temp`.
        I, Q, U = 0, 1, 2
        for i, (freq, _) in enumerate(zip(freqs, weights)):
            # apply the break frequency
            if freq < self.__freq_break:
                # TODO: this will calculate the HD17 scaling at the break
                # frequency each time a frequency below 10 GHz is requested.
                # Could store this to save recalculating it each time.
                scaling_i, scaling_p = self.evaluate_mbb_scaling(freq)
            else:
                scaling_i, scaling_p = self.evaluate_hd17_model_scaling(freq)
            temp[I, :] = self.I_ref.value
            temp[Q, :] = self.Q_ref.value
            temp[U, :] = self.U_ref.value
            temp[I] *= scaling_i
            temp[Q] *= scaling_p
            temp[U] *= scaling_p
            if len(freqs) > 1:
                utils.trapz_step_inplace(freqs, weights, i, temp, output)
        return output << u.uK_RJ
