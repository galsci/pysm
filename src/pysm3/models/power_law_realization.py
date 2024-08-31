import healpy as hp
import numpy as np

from .. import units as u
from .. import utils
from .power_law import PowerLaw
from .template import Model


class PowerLawRealization(PowerLaw):
    def __init__(
        self,
        largescale_alm,
        freq_ref,
        amplitude_modulation_temp_alm,
        amplitude_modulation_pol_alm,
        small_scale_cl,
        largescale_alm_pl_index,
        small_scale_cl_pl_index,
        nside,
        amplitude_modulation_beta_alm=None,
        galplane_fix=None,
        max_nside=None,
        seeds=None,
        synalm_lmax=None,
        has_polarization=True,
        map_dist=None,
    ):
        """PowerLaw model with stochastic small scales

        Small scale fluctuations in the templates and the spectral index
        are generated on the fly based on the input power spectra, then
        added to deterministic large scales.

        Parameters
        ----------
        largescale_alm, largescale_alm_pl_index: `pathlib.Path`
            Paths to the Alm expansion of the template IQU maps and the spectral index
            Templates are assumed to be in logpoltens formalism, units refer to
            the unit of the maps when transformed back to IQU maps.
        freq_ref: Quantity or string
            Reference frequencies at which the intensity and polarization
            templates are defined. They should be a astropy Quantity object
            or a string (e.g. "1500 MHz") compatible with GHz.
        amplitude_modulation_temp_alm, amplitude_modulation_pol_alm: `pathlib.Path`
            Paths to the Alm expansion of the modulation maps used to rescale the small scales
            to make them more un-uniform.
        amplitude_modulation_beta_alm: `pathlib.Path`
            Potentially, a different modulation map to be used for beta
        galplane_fix: `pathlib.Path`
            Set to None to skip the galactic plane fix in order to save some memory and
            computing time. Used to replace the galactic emission
            inside the GAL 070 Planck mask to a precomputed map. This is used to avoid
            excess power in the full sky spectra due to the generated small scales being
            too strong on the galactic plane.
        small_scale_cl, small_scale_cl_pl_index: `pathlib.Path`
            Paths to the power spectra of the small scale fluctuations for logpoltens iqu and
            the spectral index
        nside: int
            Resolution parameter at which this model is to be calculated.
        seeds: list of ints
            List of seeds used for generating the small scales, first is used for the template,
            the second for the spectral index. If None, it uses random seeds.
        synalm_lmax: int
            Lmax of Synalm for small scales generation, by default it is 3*nside-1,
            with a maximum of 16384.
        map_dist: Map distribution
            Unsupported, this class doesn't support MPI Parallelization
        """
        Model.__init__(self, nside=nside, max_nside=max_nside, map_dist=map_dist)
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
            if amplitude_modulation_beta_alm is not None:
                self.modulate_alm.append(
                    self.read_alm(amplitude_modulation_beta_alm, has_polarization=False)
                )
            self.small_scale_cl = self.read_cl(small_scale_cl).to(u.uK_RJ**2)

        if galplane_fix is not None:
            self.galplane_fix_map = self.read_map(galplane_fix, field=(0, 1, 2, 3))
        else:
            self.galplane_fix_map = None

        self.largescale_alm_pl_index = self.read_alm(
            largescale_alm_pl_index,
            has_polarization=False,
        ).to(u.dimensionless_unscaled)
        self.small_scale_cl_pl_index = self.read_cl(small_scale_cl_pl_index).to(
            u.dimensionless_unscaled
        )
        self.nside = int(nside)
        (
            self.I_ref,
            self.Q_ref,
            self.U_ref,
            self.pl_index,
        ) = self.draw_realization(synalm_lmax, seeds)

    def draw_realization(self, synalm_lmax=None, seeds=None):

        if seeds is None:
            seeds = (None, None)

        if synalm_lmax is None:
            synalm_lmax = int(min(16384, 2.5 * self.nside))

        output_lmax = int(min(synalm_lmax, 2.5 * self.nside))

        np.random.seed(seeds[0])

        alm_small_scale = hp.synalm(
            list(self.small_scale_cl.value),
            lmax=synalm_lmax,
            new=True,
        )

        alm_small_scale = [
            hp.almxfl(each, np.ones(output_lmax + 1)) for each in alm_small_scale
        ]
        map_small_scale = hp.alm2map(alm_small_scale, nside=self.nside)

        # need later for beta
        modulate_map_I = hp.alm2map(self.modulate_alm[0].value, self.nside)

        map_small_scale[0] *= modulate_map_I
        map_small_scale[1:] *= hp.alm2map(self.modulate_alm[1].value, self.nside)

        if len(self.modulate_alm) == 3:
            modulate_map_I = hp.alm2map(self.modulate_alm[2].value, self.nside)

        map_small_scale += hp.alm2map(
            self.template_largescale_alm.value,
            nside=self.nside,
        )

        output_IQU = (
            utils.log_pol_tens_to_map(map_small_scale)
            * self.template_largescale_alm.unit
        )

        if self.galplane_fix_map is not None:
            output_IQU *= hp.ud_grade(self.galplane_fix_map[3].value, self.nside)
            output_IQU += (
                hp.ud_grade(
                    self.galplane_fix_map[:3].value
                    * (1 - self.galplane_fix_map[3].value),
                    self.nside,
                )
                * self.galplane_fix_map.unit
            )

        np.random.seed(seeds[1])
        output_unit = np.sqrt(1 * self.small_scale_cl_pl_index.unit).unit
        alm_small_scale = hp.synalm(
            self.small_scale_cl_pl_index.value,
            lmax=synalm_lmax,
            new=True,
        )

        alm_small_scale = hp.almxfl(alm_small_scale, np.ones(output_lmax + 1))
        pl_index = hp.alm2map(alm_small_scale, nside=self.nside) * output_unit
        pl_index *= modulate_map_I
        pl_index += (
            hp.alm2map(
                self.largescale_alm_pl_index.value,
                nside=self.nside,
            )
            * output_unit
        )
        pl_index -= 3.1 * u.dimensionless_unscaled

        # Fixed values for comparison with s4
        # pl_index = -3.1 * u.dimensionless_unscaled

        return (output_IQU[0], output_IQU[1], output_IQU[2], pl_index)
