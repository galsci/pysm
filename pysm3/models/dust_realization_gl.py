import numpy as np
import healpy as hp

from .. import utils
from .dust_realization import ModifiedBlackBodyRealization


class ModifiedBlackBodyRealizationGL(ModifiedBlackBodyRealization):
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
        fwhm,
        nside,
        galplane_fix=None,
        seeds=None,
        synalm_lmax=None,
        has_polarization=True,
        map_dist=None,
    ):
        super().__init__(
            largescale_alm=largescale_alm,
            freq_ref=freq_ref,
            amplitude_modulation_temp_alm=amplitude_modulation_temp_alm,
            amplitude_modulation_pol_alm=amplitude_modulation_pol_alm,
            small_scale_cl=small_scale_cl,
            largescale_alm_mbb_index=largescale_alm_mbb_index,
            small_scale_cl_mbb_index=small_scale_cl_mbb_index,
            largescale_alm_mbb_temperature=largescale_alm_mbb_temperature,
            small_scale_cl_mbb_temperature=small_scale_cl_mbb_temperature,
            fwhm=fwhm,
            nside=nside,
            galplane_fix=galplane_fix,
            seeds=seeds,
            synalm_lmax=synalm_lmax,
            has_polarization=has_polarization,
            map_dist=map_dist,
        )

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

        lmax = min(synalm_lmax, 3 * self.nside - 1)
        alm_small_scale = np.array(
            [hp.almxfl(each, np.ones(lmax)) for each in alm_small_scale]
        )
        nlon, nlat = utils.lmax2nlon(lmax), utils.lmax2nlat(lmax)
        map_small_scale = utils.gl_alm2map(alm_small_scale, lmax, nlon=nlon, nlat=nlat)

        # need later for beta, Td
        modulate_map_I = utils.gl_alm2map(
            self.modulate_alm[0].value, lmax, nlon=nlon, nlat=nlat
        )[0]

        map_small_scale[0] *= modulate_map_I
        map_small_scale[1:] *= utils.gl_alm2map(
            self.modulate_alm[1].value, lmax, nlon=nlon, nlat=nlat
        )[0]

        map_small_scale += utils.gl_alm2map(
            self.template_largescale_alm.value, lmax, nlon=nlon, nlat=nlat
        )

        output_IQU = (
            utils.log_pol_tens_to_map(map_small_scale)
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
            alm_small_scale = hp.synalm(input_cl.value, lmax=synalm_lmax, new=True)

            alm_small_scale = hp.almxfl(alm_small_scale, np.ones(lmax))
            output[key] = (
                utils.gl_alm2map(alm_small_scale, lmax, nlon=nlon, nlat=nlat)[0]
                * output_unit
            )
            output[key] *= modulate_map_I
            output[key] += (
                utils.gl_alm2map(
                    getattr(self, f"largescale_alm_{key}").value,
                    lmax,
                    nlon=nlon,
                    nlat=nlat,
                )[0]
                * output_unit
            )

        return (
            output_IQU[0],
            output_IQU[1],
            output_IQU[2],
            output["mbb_index"],
            output["mbb_temperature"],
        )
