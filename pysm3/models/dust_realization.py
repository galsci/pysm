import numpy as np
import healpy as hp

from .. import units as u
from .. import utils
from .dust import ModifiedBlackBody


class ModifiedBlackBodyRealization(ModifiedBlackBody):
    """This is a model for modified black body emission with
    small scales generated on the fly.
    """

    def __init__(
        self,
        largescale_alm,
        freq_ref,
        amplitude_modulation_temp_alm,
        amplitude_modulation_pol_alm,
        small_scale_cl,
        largescale_alm_mbb_index,
        small_scale_cl_mbb_index,
        largescale_alm_mbb_temperature,
        small_scale_cl_mbb_temperature,
        nside,
        has_polarization=True,
        unit_mbb_temperature=None,
        map_dist=None,
    ):
        """This function initializes the modified black body model.

        The initialization of this model consists of reading in emission
        templates from file, reading in spectral parameter maps from
        file.

        Parameters
        ----------
        largescale_alm, largescale_alm_mbb_index, largescale_alm_mbb_temperature: `pathlib.Path`
            Paths to the Alm expansion of the dust template IQU maps, the spectral index
            and the dust black-body temperature.
            Templates are assumed to be in logpoltens formalism, units refer to
            the unit of the maps when transformed back to IQU maps.
        freq_ref: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        amplitude_modulation_temp_alm,
        amplitude_modulation_pol_alm,
        small_scale_cl,
        largescale_alm_mbb_index,
        small_scale_cl_mbb_index,
        largescale_alm_mbb_temperature,
        small_scale_cl_mbb_temperature,
        nside: int
            Resolution parameter at which this model is to be calculated.
        unit_mbb_temperature=None,
        map_dist=None,
        """
        self.nside = nside
        assert nside is not None
        self.map_dist = map_dist
        self.has_polarization = has_polarization

        self.freq_ref_I = u.Quantity(freq_ref).to(u.GHz)
        self.freq_ref_P = self.freq_ref_I
        with u.set_enabled_equivalencies(u.cmb_equivalencies(self.freq_ref_I)):

            self.template_largescale_alm = self.read_alm(
                largescale_alm, has_polarization=self.has_polarization
            ).to(u.uK_RJ)
            self.modulate_alm = [
                self.read_alm(each, has_polarization=False)
                for each in [
                    amplitude_modulation_temp_alm,
                    amplitude_modulation_pol_alm,
                ]
            ]
            self.small_scale_cl = self.read_cl(small_scale_cl).to(u.uK_RJ ** 2)
        self.nside = int(nside)
        (
            self.I_ref,
            self.Q_ref,
            self.U_ref,
            self.mbb_index,
            self.mbb_temperature,
        ) = self.draw_realization(seed=8192)

    def draw_realization(self, seed=None):

        synalm_lmax = 8192 * 2  # for reproducibility

        np.random.seed(seed)

        alm_log_pol_tens_small_scale = hp.synalm(
            list(self.small_scale_cl.value)
            + [np.zeros_like(self.small_scale_cl[0])] * 3,
            lmax=synalm_lmax,
            new=True,
        )

        alm_log_pol_tens_small_scale = [
            hp.almxfl(each, np.ones(3 * self.nside - 1))
            for each in alm_log_pol_tens_small_scale
        ]
        map_log_pol_tens_small_scale = hp.alm2map(
            alm_log_pol_tens_small_scale, nside=self.nside
        )
        map_log_pol_tens_small_scale[0] *= hp.alm2map(
            self.modulate_alm[0].value, self.nside
        )
        map_log_pol_tens_small_scale[1:] *= hp.alm2map(
            self.modulate_alm[1].value, self.nside
        )

        I_ref, Q_ref, U_ref = (
            utils.log_pol_tens_to_map(
                map_log_pol_tens_small_scale
                + hp.alm2map(
                    self.template_largescale_alm.value,
                    nside=self.nside,
                )
            )
            * self.template_largescale_alm.unit
        ) * 0.911  # includes color correction
        # See https://github.com/galsci/pysm/issues/99

        mbb_index = 1.48 * u.dimensionless_unscaled
        mbb_temperature = 19.6 * u.K
        return I_ref, Q_ref, U_ref, mbb_index, mbb_temperature
