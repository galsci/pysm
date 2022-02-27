import numpy as np
import healpy as hp

from .. import units as u
from .. import utils
from .dust import ModifiedBlackBody


class ModifiedBlackBodyRealization(ModifiedBlackBody):

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
        seeds=None,
        synalm_lmax=None,
        has_polarization=True,
        map_dist=None,
    ):
        """Modified Black Body model with stochastic small scales

        Small scale fluctuations in the templates, the spectral index and the black body
        temperature are generated on the fly based on the input power spectra, then
        added to deterministic large scales.

        In order to reproduce `d10`, set seeds to [8192,777,888] and synalm_max to 16384,
        either by passing arguments to the class constructor or by creating a configuration
        file based on the `d11` parameters in `data/presets.cfg` uncommenting the final
        section labelled "Configuration for reproducing d10"

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
        amplitude_modulation_temp_alm, amplitude_modulation_pol_alm: `pathlib.Path`
            Paths to the Alm expansion of the modulation maps used to rescale the small scales
            to make them more un-uniform, they are derived from highly smoothed input emission.
        small_scale_cl, small_scale_cl_mbb_index, small_scale_cl_mbb_temperature: `pathlib.Path`
            Paths to the power spectra of the small scale fluctuations for logpoltens iqu and
            the black body spectral index and temperature
        nside: int
            Resolution parameter at which this model is to be calculated.
        seeds: list of ints
            List of seeds used for generating the small scales, first is used for the template,
            the second for the spectral index, the third for the black body temperature.
            In order to reproduce `d10`, set them to [8192,777,888], if None, it uses random seeds.
        synalm_lmax: int
            Lmax of Synalm for small scales generation, by default it is 3*nside-1,
            with a maximum of 16384.
            In order to reproduce `d10`, you need to set it to 16834.
        map_dist: Map distribution
            Unsupported, this class doesn't support MPI Parallelization
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
        self.largescale_alm_mbb_index = self.read_alm(
            largescale_alm_mbb_index,
            has_polarization=False,
        ).to(u.dimensionless_unscaled)
        self.small_scale_cl_mbb_index = self.read_cl(small_scale_cl_mbb_index).to(
            u.dimensionless_unscaled
        )
        self.largescale_alm_mbb_temperature = self.read_alm(
            largescale_alm_mbb_temperature, has_polarization=False
        ).to(u.K)
        self.small_scale_cl_mbb_temperature = self.read_cl(
            small_scale_cl_mbb_temperature
        ).to(u.K ** 2)
        self.nside = int(nside)
        (
            self.I_ref,
            self.Q_ref,
            self.U_ref,
            self.mbb_index,
            self.mbb_temperature,
        ) = self.draw_realization(synalm_lmax, seeds)

    def draw_realization(self, synalm_lmax=None, seeds=None):

        if seeds is None:
            seeds = (None, None, None)

        if synalm_lmax is None:
            synalm_lmax = min(16384, 3 * self.nside - 1)

        np.random.seed(seeds[0])

        alm_small_scale = hp.synalm(
            list(self.small_scale_cl.value)
            + [np.zeros_like(self.small_scale_cl[0])] * 3,
            lmax=synalm_lmax,
            new=True,
        )

        alm_small_scale = [
            hp.almxfl(each, np.ones(min(synalm_lmax, 3 * self.nside - 1)))
            for each in alm_small_scale
        ]
        map_small_scale = hp.alm2map(alm_small_scale, nside=self.nside)

        # need later for beta, Td
        modulate_map_I = hp.alm2map(self.modulate_alm[0].value, self.nside)

        map_small_scale[0] *= modulate_map_I
        map_small_scale[1:] *= hp.alm2map(self.modulate_alm[1].value, self.nside)

        I_ref, Q_ref, U_ref = (
            utils.log_pol_tens_to_map(
                map_small_scale
                + hp.alm2map(
                    self.template_largescale_alm.value,
                    nside=self.nside,
                )
            )
            * self.template_largescale_alm.unit
        ) * 0.911  # includes color correction
        # See https://github.com/galsci/pysm/issues/99

        # Fixed values for comparison with d9
        # mbb_index = 1.48 * u.dimensionless_unscaled
        # mbb_temperature = 19.6 * u.K

        output = {}

        for seed, key in zip(seeds[1:], ["mbb_index", "mbb_temperature"]):
            np.random.seed(seed)
            input_cl = getattr(self, f"small_scale_cl_{key}")
            output_unit = np.sqrt(1 * input_cl.unit).unit
            alm_small_scale = hp.synalm(
                input_cl.value,
                lmax=synalm_lmax,
                new=True,
            )

            alm_small_scale = hp.almxfl(
                alm_small_scale, np.ones(min(3 * self.nside - 1, synalm_lmax + 1))
            )
            output[key] = hp.alm2map(alm_small_scale, nside=self.nside) * output_unit
            output[key] *= modulate_map_I
            output[key] += (
                hp.alm2map(
                    getattr(self, f"largescale_alm_{key}").value,
                    nside=self.nside,
                )
                * output_unit
            )

        return I_ref, Q_ref, U_ref, output["mbb_index"], output["mbb_temperature"]
