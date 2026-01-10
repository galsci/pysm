import os.path

import numpy as np
from numba import njit

import pysm3 as pysm
import pysm3.units as u

from .. import utils
from .cmb import CMBMap
from .interpolating import InterpolatingComponent
from .template import Model


@njit
def SPT_CIB_map_scaling(nu):
    """CIB maps correction based on SPT data

    Parameters
    ----------
    nu : float or np.array
        frequency in GHz

    Returns
    -------
    correction : float or np.array
        correction factor to be applied to the map
    """
    return np.sqrt(1 + (1.84 - 1) / (1 + np.exp((nu - 227) / 75)))


@njit
def y2uK_CMB(nu):
    """Compton-y distortion at a given frequency
    Parmeters:
    nu (float): frequency in GHz
    Returns:
    float: intensity variation dT_CMB in micro-Kelvin
      dT_CMB = dI_nu / (dB_nu / dT)_Tcmb
      where B_nu is the Planck function and dI_nu is the intensity distortion
    """

    h = 6.62607004e-27
    k = 1.380622e-16
    Tcmb = 2.725
    x = h * nu * 1e9 / k / Tcmb
    return 1e6 * Tcmb * (x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4)


class WebSkyCIB(InterpolatingComponent):
    """PySM component interpolating between precomputed maps"""

    def __init__(
        self,
        nside,
        websky_version="0.4",
        input_units="MJy / sr",
        max_nside=4096,
        interpolation_kind="linear",
        apply_SPT_correction=True,
        local_folder=None,
        map_dist=None,
    ):
        """Load and interpolate WebSky CIB maps

        Parameters
        ----------
        nside : nside
            target nside of the output maps
        websky_version : str
            currently only 0.4 is supported
        input_units : str
            input units string, e.g. uK_CMB, K_RJ
        max_nside : int
            maximum nside at which the input maps are available at
            `nside` can be higher than this, but then PySM will use
            `ud_grade` to create maps at higher resolution.
        interpolation_kind : str
            See the docstring of :py:class:`~pysm3.InterpolatingComponent`
        apply_SPT_correction : bool
            Apply the correction computed by comparison with the South
            Pole Telescope maps.
        local_folder : str
            Override the input maps folder
        map_dist : :py:class:`~pysm3.MapDistribution`
            Required for partial sky or MPI, see the PySM docs
        """
        self.local_folder = local_folder
        self.websky_freqs_float = [
            17.0,
            18.7,
            21.6,
            24.5,
            27.3,
            30.0,
            35.9,
            41.7,
            44.0,
            47.4,
            63.9,
            67.8,
            70.0,
            73.7,
            79.6,
            90.2,
            100,
            111,
            129,
            143,
            153,
            164,
            189,
            210,
            217,
            232,
            256,
            275,
            294,
            306,
            314,
            340,
            353,
            375,
            409,
            467,
            525,
            545,
            584,
            643,
            729,
            817,
            857,
            906,
            994,
            1080,
        ]
        self.websky_freqs = [f"{f:06.1f}" for f in self.websky_freqs_float]
        self.apply_SPT_correction = apply_SPT_correction
        super().__init__(
            path=websky_version,
            input_units=input_units,
            nside=nside,
            max_nside=max_nside,
            interpolation_kind=interpolation_kind,
            map_dist=map_dist,
        )
        self.remote_data = utils.RemoteData()

    def get_filenames(self, path):
        """Get filenames for a websky version
        For a standard interpolating component, we list files in folder,
        here we need to know the names in advance so that we can only download
        the required maps.
        """

        websky_version = path

        filenames = {
            float(str_freq): f"websky/{websky_version}/cib/cib_{str_freq}.fits"
            for str_freq in self.websky_freqs
        }

        if self.local_folder is not None:
            for freq in filenames:
                filenames[freq] = os.path.join(self.local_folder, filenames[freq])
        return filenames

    def read_map_by_frequency(self, freq):
        filename = self.remote_data.get(self.maps[freq])
        m = self.read_map_file(freq, filename)
        if self.apply_SPT_correction:
            m *= SPT_CIB_map_scaling(freq)
        return m


# radio galaxies are just like CIB, just interpolating
class WebSkyRadioGalaxies(WebSkyCIB):
    def __init__(
        self,
        nside,
        websky_version="0.4",
        input_units="Jy / sr",
        max_nside=4096,
        interpolation_kind="linear",
        apply_SPT_correction=False,
        local_folder=None,
        map_dist=None,
    ):
        super().__init__(
            nside=nside,
            websky_version=websky_version,
            input_units=input_units,
            max_nside=max_nside,
            interpolation_kind=interpolation_kind,
            apply_SPT_correction=apply_SPT_correction,
            local_folder=local_folder,
            map_dist=map_dist,
        )

    def get_filenames(self, path):
        """Get filenames for a websky version
        For a standard interpolating component, we list files in folder,
        here we need to know the names in advance so that we can only download
        the required maps.
        """

        websky_version = path

        filenames = {
            float(str_freq): f"websky/{websky_version}/radio/radio_{str_freq}.fits"
            for str_freq in self.websky_freqs
        }

        if self.local_folder is not None:
            for freq in filenames:
                filenames[freq] = os.path.join(self.local_folder, filenames[freq])
        return filenames


class SimpleSZ(Model):
    """Simple, frequency-independent SZ model using a single template map.

    This component uses a precomputed SZ template map that is independent of
    observing frequency. The same sky template is used at all frequencies and
    is retrieved via ``template_name``.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE of the output maps.
    template_name : str
        Name or key identifying the SZ template map to download/load via
        :class:`pysm3.models.utils.RemoteData`. The template is expected to be
        a single-frequency SZ map in units of uK_CMB.
    sz_type : str
        Type of SZ effect to model, either ``"kinetic"`` or ``"thermal"``.
    max_nside : int
        Maximum HEALPix NSIDE at which the input template map is available.
    map_dist : object, optional
        HEALPix map distribution helper or MPI communicator-like object used
        by the base :class:`Model` class to handle distributed maps.
    """

    def __init__(
        self,
        nside,
        template_name,
        sz_type,
        max_nside,
        map_dist=None,
    ):

        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        self.sz_type = sz_type
        self.remote_data = utils.RemoteData()
        filename = self.remote_data.get(template_name)
        self.m = self.read_map(filename, field=0, unit=u.uK_CMB)

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:

        freqs = pysm.check_freq_input(freqs)
        weights = pysm.normalize_weights(freqs, weights)

        # input map is in uK_CMB, we multiply the weights which are
        # in uK_RJ by the conversion factor of uK_CMB->uK_RJ
        # this is the equivalent of
        weights = (weights * u.uK_CMB).to_value(
            u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)
        )

        is_thermal = self.sz_type == "thermal"
        return (
            get_sz_emission_numba(freqs, weights, self.m.value, is_thermal) << u.uK_RJ
        )

        # the output of out is always 2D, (IQU, npix)


@njit(parallel=True)
def get_sz_emission_numba(freqs, weights, m, is_thermal):
    output = np.zeros((3, len(m)), dtype=np.float64)
    for i in range(len(freqs)):
        signal = m * y2uK_CMB(freqs[i]) if is_thermal else m.astype(np.float64)
        pysm.utils.trapz_step_inplace(freqs, weights, i, signal, output[0])
    return output


class WebSkyCMB(CMBMap):
    def __init__(
        self,
        nside,
        max_nside=4096,
        websky_version=0.4,
        seed=1,
        lensed=True,
        include_solar_dipole=False,
        map_dist=None,
    ):
        template_nside = 512 if nside <= 512 else 4096
        lens = "" if lensed else "un"
        soldip = "solardipole_" if include_solar_dipole else ""
        filenames = [
            utils.RemoteData().get(
                f"websky/{websky_version}/cmb/map_{pol}_{lens}"
                + f"lensed_alm_seed{seed}_{soldip}nside{template_nside}.fits"
            )
            for pol in "IQU"
        ]
        super().__init__(
            map_I=filenames[0],
            map_Q=filenames[1],
            map_U=filenames[2],
            nside=nside,
            max_nside=max_nside,
            map_dist=map_dist,
        )
