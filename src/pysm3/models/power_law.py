import numpy as np
from numba import njit

from .. import units as u
from .. import utils
from .template import Model


def _smoothing_to_radians(smoothing):
    """Convert an optional presmoothing angle to radians, 0.0 if none/zero."""
    if smoothing is None or np.all(np.asarray(smoothing) == 0):
        return 0.0
    return smoothing.to_value(u.radian)


class PowerLaw(Model):
    """This is a model for a simple power law synchrotron model."""

    def __init__(
        self,
        map_I,
        freq_ref_I,
        map_pl_index,
        nside,
        max_nside=None,
        available_nside=None,
        has_polarization=True,
        map_Q=None,
        map_U=None,
        freq_ref_P=None,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        smoothing_angle=None,
        map_dist=None,
    ):
        """This function initialzes the power law model of synchrotron
        emission.

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
            templates are defined.  They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        map_pl_index: `pathlib.Path` object
            Path to the map to be used as the power law index.
        nside: int
            Resolution parameter at which this model is to be calculated.
        max_nside: int
            Maximum resolution parameter at which this model is to be calculated.
        has_polarization: bool
            If True, the model will include polarization.
        available_nside: list of int
            List of available nside for the input maps.
        smoothing_angle : astropy.units.Quantity, optional
            Target output FWHM. When set, each amplitude template is smoothed by
            only the *differential* between this target and the presmoothing it
            already carries (read from its ``SMOOTHING_ANGLE`` header), instead
            of being smoothed by the full target. Templates whose native presmoothing
            already exceeds the target are left unchanged.
        map_dist: pysm.MapDistribution
            Distribution object used for parallel computing with MPI
        """
        super().__init__(
            nside,
            max_nside=max_nside,
            available_nside=available_nside,
            map_dist=map_dist,
        )
        if smoothing_angle is not None and map_dist is not None:
            raise NotImplementedError(
                "Differential smoothing (smoothing_angle) is not implemented "
                "for MPI-distributed maps (map_dist); build the model "
                "serially or pre-smooth the templates to the target resolution"
            )
        # do model setup
        self.is_IQU = has_polarization and map_Q is None
        self.I_ref = self.read_map(
            map_I, field=[0, 1, 2] if self.is_IQU else 0, unit=unit_I
        )
        # Record the presmoothing of each amplitude template. The angle is read
        # from the FITS header by `read_map`; grab it before splitting the IQU
        # map below, since indexing drops the header attribute.
        smoothing_I = getattr(self.I_ref, "smoothing_angle", None)
        # This does unit conversion in place so we do not copy the data
        # we do not keep the original unit because otherwise we would need
        # to make a copy of the array when we run the model
        self.I_ref <<= u.uK_RJ
        self.freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
        self.freq_ref_P = (
            None if freq_ref_P is None else u.Quantity(freq_ref_P).to(u.GHz)
        )
        self.has_polarization = has_polarization
        if self.has_polarization and map_Q is not None:
            self.Q_ref = self.read_map(map_Q, unit=unit_Q)
            self.Q_ref <<= u.uK_RJ
            self.U_ref = self.read_map(map_U, unit=unit_U)
            self.U_ref <<= u.uK_RJ
            smoothing_Q = getattr(self.Q_ref, "smoothing_angle", None)
            smoothing_U = getattr(self.U_ref, "smoothing_angle", None)
        elif self.has_polarization:  # unpack IQU map to 3 arrays
            smoothing_Q = smoothing_U = smoothing_I
            self.Q_ref = self.I_ref[1]
            self.U_ref = self.I_ref[2]
            self.I_ref = self.I_ref[0]
        else:
            smoothing_Q = smoothing_U = None
        self.pre_applied_beam = np.array(
            [
                _smoothing_to_radians(smoothing_I),
                _smoothing_to_radians(smoothing_Q),
                _smoothing_to_radians(smoothing_U),
            ]
        ) << u.radian
        try:  # input is a number
            self.pl_index = u.Quantity(map_pl_index, unit="")
        except TypeError:  # input is a path
            self.pl_index = self.read_map(map_pl_index, unit="")
        if smoothing_angle is not None:
            self.apply_differential_smoothing(smoothing_angle)

    def apply_differential_smoothing(self, smoothing_angle):
        """Smooth the amplitude templates by only the *differential* between
        the requested target beam and the presmoothing each template already
        carries, rather than by the full target beam.

        This is the core of the presmoothing feature: a template that already
        records e.g. a ``53 arcmin`` beam and is requested at ``1 deg`` is
        smoothed by ``sqrt(1deg**2 - 53arcmin**2)`` instead of a fresh ``1 deg``
        (which would double-apply roughly half of the beam). Only the
        beam-carrying amplitude maps (I / Q / U) are smoothed; the spectral
        index map is left untouched. After the call, ``pre_applied_beam``
        records the target for the smoothed templates, so applying the method
        again with the same target is a no-op.

        When the model is polarized, Q and U are components of a spin-2
        field: they are smoothed jointly with I through the TEB transform
        (the same convention as :func:`pysm3.apply_smoothing_and_coord_transform`
        on a ``(3, npix)`` map), with a per-component beam window. Smoothing
        them as independent scalar maps would apply the spin-0 transform to a
        spin-2 field, mixing E into B. The intensity-only model smooths I as a
        scalar map, which is exact for a spin-0 field.

        Raises
        ------
        NotImplementedError
            If the model was built with ``map_dist`` (MPI-distributed maps):
            differential smoothing is only implemented on the serial path.
        """
        if self.map_dist is not None:
            raise NotImplementedError(
                "Differential smoothing (smoothing_angle) is not implemented "
                "for MPI-distributed maps (map_dist); build the model "
                "serially or pre-smooth the templates to the target resolution"
            )
        if self.has_polarization:
            iqu = u.Quantity(
                np.stack([self.I_ref.value, self.Q_ref.value, self.U_ref.value]),
                self.I_ref.unit,
            )
            iqu = utils.apply_differential_smoothing(
                iqu, self.pre_applied_beam, smoothing_angle
            )
            self.I_ref, self.Q_ref, self.U_ref = iqu[0], iqu[1], iqu[2]
        else:
            self.I_ref = utils.apply_differential_smoothing(
                self.I_ref, self.pre_applied_beam[0], smoothing_angle
            )
        # the templates now carry the target beam (where it exceeded their
        # presmoothing; smaller targets cannot de-convolve), record it so a
        # second application, e.g. via Sky(smoothing_angle=...), is a no-op
        self.pre_applied_beam = np.maximum(
            self.pre_applied_beam, smoothing_angle.to(u.radian)
        )

    @u.quantity_input
    def get_emission(self, freqs: u.Quantity[u.GHz], weights=None):
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        if not self.has_polarization:
            outputs = (
                get_emission_numba_IQU(
                    freqs,
                    weights,
                    self.I_ref.value,
                    None,
                    None,
                    self.freq_ref_I.value,
                    None,
                    self.pl_index.value,
                )
                << u.uK_RJ
            )
        else:
            outputs = (
                get_emission_numba_IQU(
                    freqs,
                    weights,
                    self.I_ref.value,
                    self.Q_ref.value,
                    self.U_ref.value,
                    self.freq_ref_I.value,
                    self.freq_ref_P.value,
                    self.pl_index.value,
                )
                << u.uK_RJ
            )
        return outputs


@njit(parallel=True)
def get_emission_numba_IQU(
    freqs, weights, I_ref, Q_ref, U_ref, freq_ref_I, freq_ref_P, pl_index
):
    has_pol = Q_ref is not None
    output = np.zeros((3, len(I_ref)), dtype=np.float64)
    I, Q, U = 0, 1, 2
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
        utils.trapz_step_inplace(
            freqs,
            weights,
            i,
            I_ref.astype(np.float64) * (np.float64(freq) / freq_ref_I) ** pl_index,
            output[I],
        )
        if has_pol:
            pol_scaling = (np.float64(freq) / freq_ref_P) ** pl_index
            utils.trapz_step_inplace(freqs, weights, i, Q_ref * pol_scaling, output[Q])
            utils.trapz_step_inplace(freqs, weights, i, U_ref * pol_scaling, output[U])
    return output


class CurvedPowerLaw(PowerLaw):
    def __init__(
        self,
        map_I,
        freq_ref_I,
        map_pl_index,
        nside,
        spectral_curvature,
        freq_curve,
        max_nside=None,
        available_nside=None,
        has_polarization=True,
        map_Q=None,
        map_U=None,
        freq_ref_P=None,
        unit_I=None,
        unit_Q=None,
        unit_U=None,
        smoothing_angle=None,
        map_dist=None,
    ):
        super().__init__(
            map_I=map_I,
            freq_ref_I=freq_ref_I,
            map_pl_index=map_pl_index,
            nside=nside,
            max_nside=max_nside,
            available_nside=available_nside,
            has_polarization=has_polarization,
            map_Q=map_Q,
            map_U=map_U,
            freq_ref_P=freq_ref_P,
            unit_I=unit_I,
            unit_Q=unit_Q,
            unit_U=unit_U,
            smoothing_angle=smoothing_angle,
            map_dist=map_dist,
        )
        try:  # input is a number
            self.spectral_curvature = u.Quantity(spectral_curvature, unit="")
        except TypeError:  # input is a path
            self.spectral_curvature = self.read_map(spectral_curvature, unit="")
        self.freq_curve = u.Quantity(freq_curve).to(u.GHz)

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None):
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        if not self.has_polarization:
            outputs = (
                get_emission_numba_IQU_curved(
                    freqs,
                    weights,
                    self.I_ref.value,
                    None,
                    None,
                    self.freq_ref_I.value,
                    None,
                    self.pl_index.value,
                    self.freq_curve.value,
                    self.spectral_curvature.value,
                )
                << u.uK_RJ
            )
        else:
            outputs = (
                get_emission_numba_IQU_curved(
                    freqs,
                    weights,
                    self.I_ref.value,
                    self.Q_ref.value,
                    self.U_ref.value,
                    self.freq_ref_I.value,
                    self.freq_ref_P.value,
                    self.pl_index.value,
                    self.freq_curve.value,
                    self.spectral_curvature.value,
                )
                << u.uK_RJ
            )
        return outputs


@njit(parallel=True)
def get_emission_numba_IQU_curved(
    freqs,
    weights,
    I_ref,
    Q_ref,
    U_ref,
    freq_ref_I,
    freq_ref_P,
    pl_index,
    freq_curve,
    curvature,
):
    has_pol = Q_ref is not None
    output = np.zeros((3, len(I_ref)), dtype=np.float64)
    I, Q, U = 0, 1, 2
    for i, (freq, _weight) in enumerate(zip(freqs, weights)):
        curvature_term = np.log((np.float64(freq) / freq_curve) ** curvature)
        utils.trapz_step_inplace(
            freqs,
            weights,
            i,
            I_ref * (np.float64(freq) / freq_ref_I) ** (pl_index + curvature_term),
            output[I],
        )
        if has_pol:
            pol_scaling = (np.float64(freq) / freq_ref_P) ** (pl_index + curvature_term)
            utils.trapz_step_inplace(freqs, weights, i, Q_ref * pol_scaling, output[Q])
            utils.trapz_step_inplace(freqs, weights, i, U_ref * pol_scaling, output[U])
    return output
