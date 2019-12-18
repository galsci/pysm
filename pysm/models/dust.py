import numpy as np
import warnings
from .. import units as u
from pathlib import Path
from .template import Model
from .. import utils
import sys
from numba import njit
from astropy import constants as const
from scipy.interpolate import RectBivariateSpline
import healpy as hp 

class ModifiedBlackBody(Model):
    """ This is a model for modified black body emission.

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
        map_mbb_index,
        map_mbb_temperature,
        nside,
        has_polarization=True,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        unit_mbb_temperature=None,
        map_dist=None,
    ):
        """ This function initializes the modified black body model.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        unit_* : string or Unit
            Unit string or Unit object for all input FITS maps, if None, the input file
            should have a unit defined in the FITS header.
        freq_ref_I, freq_ref_P: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        map_mbb_index: `pathlib.Path` object or scalar value
            Path to the map to be used as the power law index for the dust
            opacity in a modified blackbody model, for a constant value use
            a float or an integer
        map_mbb_temperature: `pathlib.Path` object or scalar
            Path to the map to be used as the temperature of the dust in a
            modified blackbody model. For a constant value use a float or an
            integer
        nside: int
            Resolution parameter at which this model is to be calculated.
        """
        super().__init__(nside=nside, map_dist=map_dist)
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
        self.mbb_index = (
            self.read_map(map_mbb_index, unit="")
            if isinstance(map_mbb_index, (str, Path))
            else u.Quantity(map_mbb_index, unit="")
        )
        self.mbb_temperature = (
            self.read_map(map_mbb_temperature, unit=unit_mbb_temperature)
            if isinstance(map_mbb_temperature, (str, Path))
            else u.Quantity(map_mbb_temperature, unit=unit_mbb_temperature)
        )
        self.mbb_temperature <<= u.K
        self.nside = int(nside)

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        outputs = get_emission_numba(
            freqs.value,
            weights,
            self.I_ref.value,
            self.Q_ref.value,
            self.U_ref.value,
            self.freq_ref_I.value,
            self.freq_ref_P.value,
            self.mbb_index.value,
            self.mbb_temperature.value,
        )
        return outputs << u.uK_RJ


@njit(parallel=True)
def get_emission_numba(
    freqs,
    weights,
    I_ref,
    Q_ref,
    U_ref,
    freq_ref_I,
    freq_ref_P,
    mbb_index,
    mbb_temperature,
):
    output = np.zeros((3, len(I_ref)), dtype=I_ref.dtype)
    if len(freqs) > 1:
        temp = np.zeros((3, len(I_ref)), dtype=I_ref.dtype)
    else:
        temp = output

    I, Q, U = 0, 1, 2
    for i, (freq, weight) in enumerate(zip(freqs, weights)):
        temp[I, :] = I_ref
        temp[Q, :] = Q_ref
        temp[U, :] = U_ref
        temp[I] *= (freq / freq_ref_I) ** (mbb_index - 2.0)
        temp[Q:] *= (freq / freq_ref_P) ** (mbb_index - 2.0)
        temp[I] *= blackbody_ratio(freq, freq_ref_I, mbb_temperature)
        temp[Q:] *= blackbody_ratio(freq, freq_ref_P, mbb_temperature)
        if len(freqs) > 1:
            utils.trapz_step_inplace(freqs, weights, i, temp, output)
    return output


class DecorrelatedModifiedBlackBody(ModifiedBlackBody):
    def __init__(
        self,
        map_I=None,
        map_Q=None,
        map_U=None,
        freq_ref_I=None,
        freq_ref_P=None,
        map_mbb_index=None,
        map_mbb_temperature=None,
        nside=None,
        map_dist=None,
        mpi_comm=None,
        correlation_length=None,
    ):
        """ See parent class for other documentation.

        Parameters
        ----------
        correlation_length: float
            This number set the scale in logarithmic space for the distance in
            freuqency past which the MBB emission becomes decorrelated. For
            frequencies much much closer than this distance, the emission is
            well correlated.
        """
        super().__init__(
            map_I,
            map_Q,
            map_U,
            freq_ref_I,
            freq_ref_P,
            map_mbb_index,
            map_mbb_temperature,
            nside,
            pixel_indices=pixel_indices,
            mpi_comm=mpi_comm,
        )
        self.correlation_length = correlation_length

    def get_emission(self, freqs):
        """ Function to calculate the emission of a decorrelated modified black
        body model.
        """
        freqs = utils.check_freq_input(freqs)
        # calculate the decorrelation
        (rho_cov_I, rho_mean_I) = get_decorrelation_matrix(
            self.freq_ref_I, freqs, self.correlation_length
        )
        (rho_cov_P, rho_mean_P) = get_decorrelation_matrix(
            self.freq_ref_P, freqs, self.correlation_length
        )
        nfreqs = freqs.shape[-1]
        extra_I = np.dot(rho_cov_I, np.random.randn(nfreqs))
        extra_P = np.dot(rho_cov_P, np.random.randn(nfreqs))
        decorr = np.zeros((nfreqs, 3))
        decorr[:, 0, None] = rho_mean_I + extra_I[:, None]
        decorr[:, 1, None] = rho_mean_P + extra_P[:, None]
        decorr[:, 2, None] = rho_mean_P + extra_P[:, None]
        # apply the decorrelation to the mbb_emission
        return decorr[..., None] * super().get_emission(freqs)


@u.quantity_input(freqs=u.GHz, correlation_length=u.dimensionless_unscaled)
def frequency_decorr_model(freqs, correlation_length) -> u.dimensionless_unscaled:
    """ Function to calculate the frequency decorrelation method of
    Vansyngel+17.
    """
    log_dep = np.log(freqs[:, None] / freqs[None, :])
    return np.exp(-0.5 * (log_dep / correlation_length) ** 2)


@u.quantity_input(
    freq_constrained=u.GHz,
    freqs_constrained=u.GHz,
    correlation_length=u.dimensionless_unscaled,
)
def get_decorrelation_matrix(
    freq_constrained, freqs_unconstrained, correlation_length
) -> u.dimensionless_unscaled:
    """ Function to calculate the correlation matrix between observed
    frequencies. This model is based on the proposed model for decorrelation
    of Vanyngel+17. The proposed frequency covariance matrix in this paper
    is implemented, and a constrained Gaussian realization for the unobserved
    frequencies is calculated.

    Notes
    -----
    For a derivation see 1109.0286.

    Parameters
    ----------
    freq_constrained: float
        Reference frequency.
    freqs_unconstrained: ndarray
        Frequencies at which to calculate the correlation matrix.
    correlation_length: float
         Parameter controlling the structure of the frequency covariance matrix.

    Returns
    -------
    ndarray
        Frequency covariance matrix used to calculate a constrained realization.
    """
    assert correlation_length >= 0
    assert isinstance(freqs_unconstrained, np.ndarray)
    freq_constrained = utils.check_freq_input(freq_constrained)
    freqs_all = np.insert(freqs_unconstrained, 0, freq_constrained)
    indref = np.where(freqs_all == freq_constrained)
    corrmatrix = frequency_decorr_model(freqs_all, correlation_length)
    rho_inv = invert_safe(corrmatrix)
    rho_uu = np.delete(np.delete(rho_inv, indref, axis=0), indref, axis=1)
    rho_uu = invert_safe(rho_uu)
    rho_inv_cu = rho_inv[:, indref]
    rho_inv_cu = np.transpose(np.array([np.delete(rho_inv_cu, indref)]))
    # get eigenvalues, w, and eigenvectors in matrix, v.
    rho_uu_w, rho_uu_v = np.linalg.eigh(rho_uu)
    # reconstruct covariance matrix using only positive eigenvalues. Take
    # square root as we use this to draw directly the pixels (sigma).
    evals = np.diag(np.sqrt(np.maximum(rho_uu_w, np.zeros_like(rho_uu_w))))
    rho_covar = np.dot(rho_uu_v, np.dot(evals, np.transpose(rho_uu_v)))
    rho_mean = -np.dot(rho_uu, rho_inv_cu)
    return (rho_covar, rho_mean)


def invert_safe(matrix):
    """Function to safely invert almost positive definite matrix.

    Parameters
    ----------
    matrix: ndarray
        matrix to invert.

    Returns
    -------
    ndaray
        inverted matrix.
    """
    mb = matrix.copy()
    w_ok = False
    while not w_ok:
        w, v = np.linalg.eigh(mb)
        wmin = np.min(w)
        if wmin > 0:
            w_ok = True
        else:
            mb += np.diag(2.0 * np.max([1e-14, -wmin]) * np.ones(len(mb)))
    winv = 1.0 / w
    return np.dot(v, np.dot(np.diag(winv), np.transpose(v)))


@njit
def blackbody_ratio(freq_to, freq_from, temp):
    """ Function to calculate the flux ratio between two frequencies for a
    blackbody at a given temperature.

    Parameters
    ----------
    freq_to: float
        Frequency to which to scale assuming black body SED.
    freq_from: float
        Frequency from which to scale assuming black body SED.
    temp: float
        Temperature of the black body.

    Returns
    -------
    float
        Black body ratio between `freq_to` and `freq_from` at temperature
        `temp`.
    """
    return blackbody_nu(freq_to, temp) / blackbody_nu(freq_from, temp)


h = const.h.value
c = const.c.value
k_B = const.k_B.value


@njit
def blackbody_nu(freq, temp):
    """Calculate blackbody flux per steradian, :math:`B_{\\nu}(T)`.

    .. note::

        Use `numpy.errstate` to suppress Numpy warnings, if desired.

    .. warning::

        Output values might contain ``nan`` and ``inf``.

    Parameters
    ----------
    in_x : number, array-like, or `~astropy.units.Quantity`
        Frequency, wavelength, or wave number.
        If not a Quantity, it is assumed to be in Hz.

    temperature : number, array-like, or `~astropy.units.Quantity`
        Blackbody temperature.
        If not a Quantity, it is assumed to be in Kelvin.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Blackbody monochromatic flux in
        :math:`erg \\; cm^{-2} s^{-1} Hz^{-1} sr^{-1}`.

    Raises
    ------
    ValueError
        Invalid temperature.

    ZeroDivisionError
        Wavelength is zero (when converting to frequency).

    """

    log_boltz = h * freq * 1e9 / (k_B * temp)
    boltzm1 = np.expm1(log_boltz)

    # Calculate blackbody flux
    bb_nu = 2.0 * h * (freq * 1e9) ** 3 / (c ** 2 * boltzm1)
    # flux = bb_nu.to(FNU, u.spectral_density(freq * 1e9))

    return bb_nu


class HensleyDraine2017(Model):
    """ This is a model for modified black body emission.

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
        map_mbb_index,
        map_mbb_temperature,
        nside,
        has_polarization=True,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        unit_mbb_temperature=None,
        map_dist=None,
        mpi_comm=None,
        f_fe=None,
        f_car=None,
        rnd_uval=True,
        seed=None
    ):
        """ This function initializes the Hensley-Draine 2017 model.

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
        map_mbb_index: `pathlib.Path` object
            Path to the map to be used as the power law index for the dust
            opacity in a modified blackbody model.
        map_mbb_temperature: `pathlib.Path` object
            Path to the map to be used as the temperature of the dust in a
            modified blackbody model.
        nside: int
            Resolution parameter at which this model is to be calculated.
        f_fe: float
            Fractional composition of grain population with iron inclusions.
        f_car: float
            Fractional composition of grain population in carbonaceous grains.
        rnd_uval: bool (optional, default=True)
            Decide whether to draw a random realization of the ISRF.
        seed: int
            Number used to seed RNG for `uval`.
        """
        super().__init__(nside=nside, map_dist=map_dist)
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
        self.mbb_index = self.read_map(map_mbb_index, unit="")
        self.mbb_temperature = self.read_map(
            map_mbb_temperature, unit=unit_mbb_temperature
        )
        self.mbb_temperature <<= u.K
        self.nside = int(nside)

        self.f_fe = f_fe
        self.f_car = f_car
        self.f_sil = 1. - f_fe

        # break frequency below which model is not valid.
        self.__freq_break = 10. * u.GHz

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

        uvec = np.arange(-3., 5.01, 0.1) * u.dimensionless_unscaled

        # The tabulated data is nu * I_nu / N_H, where N_H is the
        # number of hydrogen atoms per cm^2. Therefore the units of
        # the tabulated data are erg / s / sr. 
        sil_data_i = (sil_data[:, 3 : 84] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)
        silfe_data_i = (silfe_data[:, 3 : 84] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)
        car_data_i = (car_data[:, 3 : 84] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)

        sil_data_p = (sil_data[:, 84 : 165] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)
        silfe_data_p = (silfe_data[:, 84 : 165] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)
        car_data_p = (car_data[:, 84 : 165] * u.erg / u.s / u.sr / wav[:, None].to(u.Hz, equivalencies=u.spectral())).to(u.Jy / u.sr * u.cm ** 2)

        # interpolate the pre-computed solutions for the emissivity as a
        # function of grain 4 composition F_fe, Fcar, and field strenth U,
        # to get emissivity as a function of (U, wavelength).
        # Note that the spline, when evaluated, returns a unitless numpy array.
        # We will later ignore the cm^2 in the unit, since this does not affect
        # the outcome, and prevents the conversion between uK_RJ and Jy / sr
        # in astropy.
        assert sil_data_i.unit == u.Jy / u.sr * u.cm ** 2
        self.sil_i = RectBivariateSpline(uvec, wav, sil_data_i.T)

        assert silfe_data_i.unit == u.Jy / u.sr * u.cm ** 2
        self.car_i = RectBivariateSpline(uvec, wav, car_data_i.T)

        assert silfe_data_i.unit == u.Jy / u.sr * u.cm ** 2
        self.silfe_i = RectBivariateSpline(uvec, wav, silfe_data_i.T)

        assert sil_data_p.unit == u.Jy / u.sr * u.cm ** 2
        self.sil_p = RectBivariateSpline(uvec, wav, sil_data_p.T)

        assert car_data_p.unit == u.Jy / u.sr * u.cm ** 2
        self.car_p = RectBivariateSpline(uvec, wav, car_data_p.T)

        assert silfe_data_p.unit == u.Jy / u.sr * u.cm ** 2
        self.silfe_p = RectBivariateSpline(uvec, wav, silfe_data_p.T)

        #now draw the random realisation of uval if draw_uval = true
        if rnd_uval:
            T_mean = self.read_map("pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits", unit="K",  field=3)
            T_std = self.read_map("pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits", unit="K", field=5)
            beta_mean = self.read_map("pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits", unit="", field=6)
            beta_std = self.read_map("pysm_2/COM_CompMap_dust-commander_0256_R2.00.fits", unit="", field=8)
            #draw the realisations

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
            self.uval = hp.ud_grade(np.clip((4. + beta) * np.log10(T / np.mean(T)), -3., 5.), nside_out = nside)
        elif not rnd_uval:
            # I think this needs filling in for case when ISRF is not
            # a random realization. What should the default be? Could
            # choose a single value corresponding to T=20K, beta_d=1.54?
            pass
        else:
            print(
                """Hensley_Draine_2017 model selected, but draw_uval not set.
                Set 'draw_uval' to True or False."""
            )

        # compute the SED at the reference frequencies of the input templates.
        lambda_ref_i = self.freq_ref_I.to(u.um, equivalencies=u.spectral())
        lambda_ref_p = self.freq_ref_P.to(u.um, equivalencies=u.spectral())
        self.i_sed_at_nu0 = (self.f_sil * self.sil_i.ev(self.uval, lambda_ref_i) \
            + self.f_car * self.car_i.ev(self.uval, lambda_ref_i) + \
                self.f_fe * self.silfe_i.ev(self.uval, lambda_ref_i)) * u.Jy / u.sr

        self.p_sed_at_nu0 = (self.f_sil * self.sil_p.ev(self.uval, lambda_ref_p) \
            + self.f_car * self.car_p.ev(self.uval, lambda_ref_p) + \
                self.f_fe * self.silfe_p.ev(self.uval, lambda_ref_p)) * u.Jy / u.sr

    @u.quantity_input
    def evaluate_hd17_model_scaling(self, freqs: u.GHz):
        """ Method to 
        """
        # interpolation over pre-computed model is done in microns, so first convert
        # to microns.
        print(freqs)
        lambdas = freqs.to(u.um, equivalencies=u.spectral())[:, None]
        # evaluate the SED, which is currently does the scaling assuming Jy/sr.
        # uval is unitless, and lambdas are un microns.
        scaling_i = (self.f_sil * self.sil_i.ev(self.uval, lambdas) + self.f_car * self.car_i.ev(self.uval, lambdas) + self.f_fe * self.silfe_i.ev(self.uval, lambdas)) / self.i_sed_at_nu0
        scaling_p = (self.f_sil * self.sil_p.ev(self.uval, lambdas) + self.f_car * self.car_p.ev(self.uval, lambdas) + self.f_fe * self.silfe_p.ev(self.uval, lambdas) ) / self.p_sed_at_nu0
        # scaling_i, and scaling_p are unitless scaling factors. However the scaling
        # does have the assumption of Jy / sr in the output map. We now account for
        # this by multiplying by the ratio of unit conversions from Jy / sr to uK_RJ
        # at the observed frequencies compared to the reference frequencies in
        # temperature and polarization.
        scaling_i *= ((u.Jy / u.sr).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs)) / (u.Jy / u.sr).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(self.freq_ref_I)))[:, None]
        scaling_p *= ((u.Jy / u.sr).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs)) / (u.Jy / u.sr).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(self.freq_ref_P)))[:, None]
        return scaling_i, scaling_p

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, **kwargs) -> u.uK_RJ:
        """
        This function calculates the model of Hensley and Draine 2017 for
        the emission of a mixture of silicate, cabonaceous, and silicate
        grains with iron inclusions.

        Parameters
        ----------
        freqs: float
            Frequencies in GHz at which to evaluate the model.

        Returns
        -------
        ndarray
            Maps of T, Q, U at frequencies `freqs`.

        """
        if ('use_bandpass' in kwargs) and (kwargs['use_bandpass']):
            return np.zeros((3, len(self.I_ref)))

        # The HD17 model relies on interpolated data, which is only valid
        # above frequencies around 10 GHz. Therefore, if a frequency below
        # this is requrested, we use a simple MBB law with beta=1.54. 
        # The higher end of the interpolation is well above what would be
        # required in a CMB experiment.
        idx = freqs > self.__freq_break
        freqs_below_cutoff = freqs[np.invert(idx)]
        freqs  = freqs[idx]

        # Evaluate the model scaling, this is returned in uK_RJ
        scaling_i, scaling_p = self.evaluate_hd17_model_scaling(freqs)

        # Handle the frequencies below the 10 GHz cutoff.
        if freqs_below_cutoff.size != 0:
            # At these frequencies dust is largely irrelevant, and so we just 
            # use a Rayleigh-Jeans model with constant spectral index of 1.54 
            # for simplicity.
            RJ_factor = (freqs_below_cutoff / self.__freq_break) ** 1.54

            #calculate the HD17  model at the break frequency.
            scaling_i_at_cutoff, scaling_p_at_cutoff = self.evaluate_hd17_model_scaling(np.array([self.__freq_break.value]) * u.GHz)

            # rescale the HD17 model at the break frequency using MBB for freqs < 10 GHz.
            scaling_i_below_cutoff = RJ_factor[:, None] * scaling_i_at_cutoff
            scaling_p_below_cutoff = RJ_factor[:, None] * scaling_p_at_cutoff

            # concatenate with models evaluated at frequencies above 10 GHz. 
            scaling_i = np.concatenate((scaling_i_below_cutoff, scaling_i))
            scaling_p = np.concatenate((scaling_p_below_cutoff, scaling_p))

        return np.array([scaling_i * self.I_ref.value, scaling_p * self.Q_ref.value, scaling_p * self.U_ref.value]) * u.uK_RJ