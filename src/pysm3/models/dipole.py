import numpy as np
import healpy as hp
from astropy import constants as const

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

from .. import units as u
from .. import utils


class CMBDipole:
    """
    Simulate the CMB dipole anisotropy as a full-sky HEALPix map.

    The dipole is modeled as a temperature fluctuation due to the observer's motion
    with respect to the CMB rest frame, following the relativistic Doppler effect.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter for the output map (dimensionless).
    amp : float
        Amplitude of the dipole in micro-Kelvin (uK_CMB).
    T_cmb : float
        CMB monopole temperature in Kelvin (K_CMB).
    dip_lon : float
        Galactic longitude of the dipole direction, in degrees (deg).
    dip_lat : float
        Galactic latitude of the dipole direction, in degrees (deg).

    Returns
    -------
    dipole_map : astropy.units.Quantity
        Full-sky HEALPix map (array) of the dipole temperature anisotropy in Kelvin.

    Notes
    -----
    The dipole amplitude is calculated as:

        ΔT/T = (v/c) * cos(θ)

    where θ is the angle between the dipole direction and the line of sight.

    Reference
    ---------
    For the best-fit parameters, see Table 1 of
    "Planck intermediate results. LVII. Joint Planck LFI and HFI data processing"
    https://arxiv.org/pdf/2007.04997.pdf
    """

    @u.quantity_input
    def __init__(
        self,
        nside: int,
        amp,
        T_cmb,
        dip_lon,
        dip_lat,
        map_dist=None,
        quadrupole_correction: bool = False,  # New flag
    ):
        self.nside = nside
        self.amp = u.Quantity(amp) if not isinstance(amp, u.Quantity) else amp
        self.T_cmb = u.Quantity(T_cmb) if not isinstance(T_cmb, u.Quantity) else T_cmb
        self.dip_lat = (
            u.Quantity(dip_lat) if not isinstance(dip_lat, u.Quantity) else dip_lat
        ).to_value(u.deg)
        self.dip_lon = (
            u.Quantity(dip_lon) if not isinstance(dip_lon, u.Quantity) else dip_lon
        ).to_value(u.deg)
        self.map_dist = map_dist
        self.quadrupole_correction = quadrupole_correction  # Store flag

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ):
        """
        Return the dipole emission map, integrating over the bandpass if needed.

        Parameters
        ----------
        freqs : Quantity
            Frequency or array of frequencies (for bandpass integration).
        weights : array-like, optional
            Integration weights for the bandpass.

        Returns
        -------
        dipole_map : astropy.units.Quantity
            Full-sky HEALPix map (array) of the dipole temperature anisotropy in uK_RJ.
        """
        npix = hp.nside2npix(self.nside)
        vec = hp.ang2vec(self.dip_lon, self.dip_lat, lonlat=True)
        pix_dirs = hp.pix2vec(self.nside, np.arange(npix))
        cosθ = np.dot(vec, pix_dirs)  # cos(theta)
        δ = (self.amp / self.T_cmb).decompose()
        β = δ * (δ + 2) / (δ**2 + 2 * δ + 2)
        γ = 1 / np.sqrt(1 - β**2)  # Lorentz factor gamma

        # this is the temperature fluctuation with no quadrupole correction
        # it does not depend on the frequency
        ΔT = self.T_cmb / (γ * (1 - β * cosθ)) - self.T_cmb

        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)

        emission = []
        for freq in freqs:
            if self.quadrupole_correction:
                fx = (const.h * (freq*u.GHz).to(u.Hz) / (const.k_B * (self.T_cmb.to_value(u.K_CMB)*u.K))).decompose()
                fcor = (fx / 2) * (np.exp(fx) + 1) / (np.exp(fx) - 1)
                bt = β * cosθ
                # with quadrupole correction the temperature fluctuation depends on the frequency
                # so it is overwritten here
                ΔT_current_freq = self.T_cmb * (bt + fcor * bt**2)
                emission.append(
                    ΔT_current_freq.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz))
                )
            elif freq == 0 * u.GHz: # No quadrupole correction and 0 GHz
                emission.append(ΔT) # ΔT is already in K
            else: # No quadrupole correction and non-zero frequency
                emission.append(
                    ΔT.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz))
                )

        if len(freqs) == 1:
            result = emission[0]
        else:
            result = (
                trapezoid(
                    np.stack([e.value for e in emission]) * weights[:, None],
                    x=freqs,
                    axis=0,
                )
                * u.uK_RJ
            )

        assert not np.isnan(result).any(), "Result contains NaN values"

        return result


class CMBDipoleQuad(CMBDipole):
    def __init__(self, *args, **kwargs):
        kwargs["quadrupole_correction"] = True
        super().__init__(*args, **kwargs)
