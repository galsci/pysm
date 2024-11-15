from pathlib import Path

import numpy as np
from astropy import constants as const
from numba import njit

from .. import units as u
from .. import utils
from .template import Model


class ModifiedBlackBody(Model):
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
        freq_ref_I,
        freq_ref_P,
        map_mbb_index,
        map_mbb_temperature,
        nside,
        max_nside=None,
        available_nside=None,
        map_Q=None,
        map_U=None,
        has_polarization=True,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        unit_mbb_temperature=None,
        map_dist=None,
    ):
        """This function initializes the modified black body model.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
            If has_polarization is True and map_Q is None, assumes map_I is IQU
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
        max_nside : int
            Maximum resolution parameter at which this model is defined.
        available_nside : list of int
            List of resolution parameters at which the input maps are defined.
        """
        super().__init__(
            nside=nside,
            max_nside=max_nside,
            map_dist=map_dist,
            available_nside=available_nside,
        )
        # do model setup
        self.is_IQU = has_polarization and map_Q is None
        self.I_ref = self.read_map(
            map_I, field=[0, 1, 2] if self.is_IQU else 0, unit=unit_I
        )
        # This does unit conversion in place so we do not copy the data
        # we do not keep the original unit because otherwise we would need
        # to make a copy of the array when we run the model
        self.I_ref <<= u.uK_RJ
        self.freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
        self.has_polarization = has_polarization
        if has_polarization and map_Q is not None:
            self.Q_ref = self.read_map(map_Q, unit=unit_Q)
            self.Q_ref <<= u.uK_RJ
            self.U_ref = self.read_map(map_U, unit=unit_U)
            self.U_ref <<= u.uK_RJ
        else:  # unpack IQU map to 3 arrays
            self.Q_ref = self.I_ref[1]
            self.U_ref = self.I_ref[2]
            self.I_ref = self.I_ref[0]
        if has_polarization:
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
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        outputs = get_emission_numba(
            freqs,
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
    output = np.zeros((3, len(I_ref)), dtype=np.float64)
    temp = np.zeros((3, len(I_ref)), dtype=np.float64) if len(freqs) > 1 else output

    I, Q, U = 0, 1, 2
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
        temp[I, :] = I_ref
        temp[Q, :] = Q_ref
        temp[U, :] = U_ref
        if freq != freq_ref_I:
            # -2 because black body is in flux unit and not K_RJ
            temp[I] *= (np.float64(freq / freq_ref_I)) ** (mbb_index - 2.0)
            temp[I] *= blackbody_ratio(
                np.float64(freq), np.float64(freq_ref_I), mbb_temperature
            )
            freq_scaling_P = (np.float64(freq / freq_ref_P)) ** (
                mbb_index - 2.0
            ) * blackbody_ratio(np.float64(freq), np.float64(freq_ref_P), mbb_temperature)
            for P in [Q, U]:
                temp[P] *= freq_scaling_P
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
        max_nside=None,
        available_nside=None,
        mpi_comm=None,
        map_dist=None,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        unit_mbb_temperature=None,
        correlation_length=None,
    ):
        """See parent class for other documentation.

        Parameters
        ----------
        correlation_length: float
            This number set the scale in logarithmic space for the distance in
            freuqency past which the MBB emission becomes decorrelated. For
            frequencies much much closer than this distance, the emission is
            well correlated.
        """
        super().__init__(
            map_I=map_I,
            map_Q=map_Q,
            map_U=map_U,
            freq_ref_I=freq_ref_I,
            freq_ref_P=freq_ref_P,
            map_mbb_index=map_mbb_index,
            map_mbb_temperature=map_mbb_temperature,
            nside=nside,
            max_nside=max_nside,
            available_nside=available_nside,
            unit_I=unit_I,
            unit_Q=unit_Q,
            unit_U=unit_U,
            unit_mbb_temperature=unit_mbb_temperature,
            map_dist=map_dist,
        )
        self.correlation_length = correlation_length * u.dimensionless_unscaled

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
        """Function to calculate the emission of a decorrelated modified black
        body model.
        """
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        # calculate the decorrelation
        rho_cov_I, rho_mean_I = get_decorrelation_matrix(
            self.freq_ref_I, freqs * u.GHz, self.correlation_length
        )
        rho_cov_P, rho_mean_P = get_decorrelation_matrix(
            self.freq_ref_P, freqs * u.GHz, self.correlation_length
        )
        nfreqs = freqs.shape[-1]
        extra_I = np.dot(rho_cov_I, np.random.randn(nfreqs))
        extra_P = np.dot(rho_cov_P, np.random.randn(nfreqs))

        decorr = np.zeros((nfreqs, 3))
        decorr[:, 0, None] = rho_mean_I + extra_I[:, None]
        decorr[:, 1, None] = rho_mean_P + extra_P[:, None]
        decorr[:, 2, None] = rho_mean_P + extra_P[:, None]

        output = np.zeros((3, len(self.I_ref)), dtype=self.I_ref.dtype)
        # apply the decorrelation to the mbb_emission for each frequencies before integrating
        for i, (freq, _weight) in enumerate(zip(freqs, weights)):
            temp = decorr[..., None][i] * super().get_emission(freq * u.GHz)
            if len(freqs) > 1:
                utils.trapz_step_inplace(freqs, weights, i, temp, output)
            else:
                output = temp
        return output << u.uK_RJ


@u.quantity_input
def frequency_decorr_model(
    freqs: u.Quantity[u.GHz], correlation_length: u.Quantity[u.dimensionless_unscaled]
):
    """Function to calculate the frequency decorrelation method of
    Vansyngel+17.
    """
    log_dep = np.log(freqs[:, None] / freqs[None, :])
    return np.exp(-0.5 * (log_dep / correlation_length) ** 2)


@u.quantity_input
def get_decorrelation_matrix(
    freq_constrained: u.Quantity[u.GHz],
    freqs_unconstrained: u.Quantity[u.GHz],
    correlation_length: u.Quantity[u.dimensionless_unscaled],
):
    """Function to calculate the correlation matrix between observed
    frequencies. This model is based on the proposed model for decorrelation
    of Vansyngel+17. The proposed frequency covariance matrix in this paper
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
    freq_constrained = utils.check_freq_input(freq_constrained) * u.GHz
    freqs_all = np.insert(freqs_unconstrained, 0, freq_constrained)
    indref = np.where(freqs_all == freq_constrained)
    corrmatrix = frequency_decorr_model(freqs_all, correlation_length)
    rho_inv = invert_safe(corrmatrix.value)
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
    return rho_covar, rho_mean


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
    """Function to calculate the flux ratio between two frequencies for a
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
    return 2.0 * h * (freq * 1e9) ** 3 / (c**2 * boltzm1)
