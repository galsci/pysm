import numpy as np
import healpy as hp
import astropy.units as u


class DipoleComponent:
    """
    Simulate the CMB dipole anisotropy as a full-sky HEALPix map.

    The dipole is modeled as a temperature fluctuation due to the observer's motion
    with respect to the CMB rest frame, following the relativistic Doppler effect.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter for the output map.
    vel : float
        Observer velocity with respect to the CMB rest frame, in km/s.
    T_cmb : float
        CMB monopole temperature in Kelvin.
    dip_lon : float
        Galactic longitude of the dipole direction, in degrees.
    dip_lat : float
        Galactic latitude of the dipole direction, in degrees.

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

    def __init__(self, nside, vel, T_cmb, dip_lon, dip_lat):
        self.nside = nside
        self.vel = vel * u.km / u.s
        self.T_cmb = T_cmb * u.K
        self.dip_lon = np.deg2rad(dip_lon)
        self.dip_lat = np.deg2rad(dip_lat)

    def get_map(self):
        npix = hp.nside2npix(self.nside)
        vec = hp.ang2vec(np.pi / 2 - self.dip_lat, self.dip_lon)
        pix_dirs = hp.pix2vec(self.nside, np.arange(npix))
        cos_theta = np.dot(vec, pix_dirs)
        delta_T = (self.vel / u.c).decompose().value * self.T_cmb.value * cos_theta
        return delta_T * u.K
