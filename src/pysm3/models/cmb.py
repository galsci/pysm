import healpy as hp
import numpy as np
from scipy.special import comb, factorial

from .. import units as u
from .. import utils
from .template import Model


class CMBMap(Model):
    """Load one or a set of 3 CMB maps"""

    def __init__(
        self,
        nside,
        max_nside=None,
        map_IQU=None,
        map_I=None,
        map_Q=None,
        map_U=None,
        map_dist=None,
    ):
        """
        The input is assumed to be in `uK_CMB`

        Parameters
        ----------
        nside: int
            HEALPix N_side parameter of the input maps
        map_IQU: `pathlib.Path` object
            Path to a single IQU map
        map_I, map_Q, map_U: `pathlib.Path` object
            Paths to the maps to be used as I, Q, U templates.
        """
        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        if map_IQU is not None:
            self.map = self.read_map(map_IQU, unit=u.uK_CMB, field=(0, 1, 2))
        elif map_I is not None:
            self.map = self.read_map(map_I, unit=u.uK_CMB, field=0)
            if map_Q is not None:
                self.map = [self.map]
                for m in [map_Q, map_U]:
                    self.map.append(self.read_map(m, unit=u.uK_CMB))
                self.map = u.Quantity(self.map, unit=u.uK_CMB)
        else:
            msg = "No input map provided"
            raise (ValueError(msg))

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
        freqs = utils.check_freq_input(freqs)
        # do not use normalize weights because that includes a transformation
        # to spectral radiance and then back to RJ
        if weights is None:
            weights = np.ones(len(freqs), dtype=np.float64)

        scaling_factor = utils.bandpass_unit_conversion(
            freqs * u.GHz, weights, output_unit=u.uK_RJ, input_unit=u.uK_CMB
        )

        return u.Quantity(self.map * scaling_factor, unit=u.uK_RJ, copy=False)


"""The following code is edited from the taylens code: Naess,
S. K. and Louis, T. 2013 'Lensing simulations by Taylor expansion -
not so inefficient after all' Journal of Cosmology and Astroparticle
Physics September 2013.  Available at:
https://github.com/amaurea/taylens

"""


def simulate_tebp_correlated(cl_tebp_arr, nside, lmax, seed):
    """This generates correlated T,E,B and Phi maps"""
    np.random.seed(seed)
    alms = hp.synalm(cl_tebp_arr, lmax=lmax, new=True)
    aphi = alms[-1]
    acmb = alms[0:-1]
    # Set to zero above map resolution to avoid aliasing
    beam_cut = np.ones(3 * nside)
    for ac in acmb:
        hp.almxfl(ac, beam_cut, inplace=True)
    cmb = np.array(hp.alm2map(acmb, nside, pol=True))
    return cmb, aphi


def taylor_interpol_iter(m, pos, order=3, verbose=False, lmax=None):
    """Given a healpix map m[npix], and a set of positions
    pos[{theta,phi},...], evaluate the values at those positions using
    harmonic Taylor interpolation to the given order (3 by
    default). Successively yields values for each cumulative order up
    to the specified one. If verbose is specified, it will print
    progress information to stderr.

    """
    nside = hp.npix2nside(m.size)
    if lmax is None:
        lmax = 3 * nside
    # Find the healpix pixel centers closest to pos,
    # and our deviation from these pixel centers.
    ipos = hp.ang2pix(nside, pos[0], pos[1])
    pos0 = np.array(hp.pix2ang(nside, ipos))
    dpos = pos[:2] - pos0
    # Take wrapping into account
    bad = dpos[1] > np.pi
    dpos[1, bad] = dpos[1, bad] - 2 * np.pi
    bad = dpos[1] < -np.pi
    dpos[1, bad] = dpos[1, bad] + 2 * np.pi

    # Since healpix' dphi actually returns dphi/sintheta, we choose
    # to expand in terms of dphi*sintheta instead.
    dpos[1] *= np.sin(pos0[0])
    del pos0

    # We will now Taylor expand our healpix field to
    # get approximations for the values at our chosen
    # locations. The structure of this section is
    # somewhat complicated by the fact that alm2map_der1 returns
    # two different derivatives at the same time.
    derivs = [[m]]
    res = m[ipos]
    yield res
    for o in range(1, order + 1):
        # Compute our derivatives
        derivs2 = [None for i in range(o + 1)]
        used = [False for i in range(o + 1)]
        # Loop through previous level in steps of two (except last)
        if verbose:
            print("order %d" % o)
        for i in range(o):
            # Each alm2map_der1 provides two derivatives, so avoid
            # doing double work.
            if i < o - 1 and i % 2 == 1:
                continue
            a = hp.map2alm(derivs[i], use_weights=True, lmax=lmax, iter=0)
            derivs[i] = None
            dtheta, dphi = hp.alm2map_der1(a, nside, lmax=lmax)[-2:]
            derivs2[i : i + 2] = [dtheta, dphi]
            del a, dtheta, dphi
            # Use these to compute the next level
            for j in range(i, min(i + 2, o + 1)):
                if used[j]:
                    continue
                N = comb(o, j) / factorial(o)
                res += N * derivs2[j][ipos] * dpos[0] ** (o - j) * dpos[1] ** j
                used[j] = True
                # If we are at the last order, we don't need to waste memory
                # storing the derivatives any more
                if o == order:
                    derivs2[j] = None
        derivs = derivs2
        yield res


def offset_pos(ipos, dtheta, dphi, pol=False, geodesic=False):
    """Offsets positions ipos on the sphere by a unit length step along
    the gradient dtheta, dphi/sintheta, taking the curvature of the
    sphere into account. If pol is passed, also computes the cos and
    sin of the angle by which (Q,U) must be rotated to take into
    account the change in local coordinate system.

    If geodesic is passed, a quick and dirty, but quite accurate,
    approximation is used.

    Uses the memory of 2 maps (4 if pol) (plus that of the input
    maps).

    """
    opos = np.zeros(ipos.shape)
    orot = np.zeros(ipos.shape) if pol and not geodesic else None
    if not geodesic:
        # Loop over chunks in order to conserve memory
        step = 0x10000
        for i in range(0, ipos.shape[1], step):
            small_opos, small_orot = offset_pos_helper(
                ipos[:, i : i + step], dtheta[i : i + step], dphi[i : i + step], pol
            )
            opos[:, i : i + step] = small_opos
            if pol:
                orot[:, i : i + step] = small_orot
    else:
        opos[0] = ipos[0] + dtheta
        opos[1] = ipos[1] + dphi / np.sin(ipos[0])
        opos = fixang(opos)
    return opos, orot


def offset_pos_helper(ipos, dtheta, dphi, pol):
    grad = np.array((dtheta, dphi))
    dtheta, dphi = None, None
    d = np.sum(grad**2, 0) ** 0.5
    grad /= d
    cosd, sind = np.cos(d), np.sin(d)
    cost, sint = np.cos(ipos[0]), np.sin(ipos[0])
    ocost = cosd * cost - sind * sint * grad[0]
    osint = (1 - ocost**2) ** 0.5
    ophi = ipos[1] + np.arcsin(sind * grad[1] / osint)
    if not pol:
        return np.array([np.arccos(ocost), ophi]), None
    A = grad[1] / (sind * cost / sint + grad[0] * cosd)
    nom1 = grad[0] + grad[1] * A
    denom = 1 + A**2
    cosgam = 2 * nom1**2 / denom - 1
    singam = 2 * nom1 * (grad[1] - grad[0] * A) / denom
    return np.array([np.arccos(ocost), ophi]), np.array([cosgam, singam])


def fixang(pos):
    """Handle pole wraparound."""
    a = np.array(pos)
    bad = np.where(a[0] < 0)
    a[0, bad] = -a[0, bad]
    a[1, bad] = a[1, bad] + np.pi
    bad = np.where(a[0] > np.pi)
    a[0, bad] = 2 * np.pi - a[0, bad]
    a[1, bad] = a[1, bad] + np.pi
    return a


def apply_rotation(m, rot):
    """Update Q,U components in polarized map by applying the rotation
    rot, represented as [cos2psi,sin2psi] per pixel. Rot is one of the
    outputs from offset_pos.

    """
    if len(m) < 3:
        return m
    if rot is None:
        return m
    m = np.asarray(m)
    res = m.copy()
    res[1] = rot[0] * m[1] - rot[1] * m[2]
    res[2] = rot[1] * m[1] + rot[0] * m[2]
    return m


class CMBLensed(CMBMap):
    # intherit from CMBMap so we get the `get_emission` method
    def __init__(
        self,
        nside,
        cmb_spectra,
        max_nside=None,
        cmb_seed=None,
        apply_delens=False,
        delensing_ells=None,
        map_dist=None,
    ):
        """Lensed CMB

        Takes an input unlensed CMB and lensing spectrum from CAMB and uses
        Taylens to apply lensing, it optionally simulates delensing by
        suppressing the lensing power at specific scales with the user
        provided `delensing_ells`.

        Parameters
        ----------

        cmb_spectra : path
            Input text file from CAMB, spectra unlensed
        cmb_seed : int
            Numpy random seed for synfast, set to None for a random seed
        apply_delens : bool
            If true, simulate delensing with taylens
        delensing_ells : path
            Space delimited file with ells in the first columns and suppression
            factor (1 for no suppression) in the second column
        """
        try:
            super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        except ValueError:
            pass  # suppress exception about not providing any input map
        self.cmb_spectra = self.read_txt(cmb_spectra, unpack=True)
        self.cmb_seed = cmb_seed
        self.apply_delens = apply_delens
        self.delensing_ells = (
            None if delensing_ells is None else self.read_txt(delensing_ells)
        )
        self.map = u.Quantity(self.run_taylens(), unit=u.uK_CMB, copy=False)

    def run_taylens(self):
        """Returns CMB (T, Q, U) maps as a function of observing frequency, nu.

        This code is extracted from the taylens code (reference).

        :return: function -- CMB maps.
        """
        synlmax = 8 * self.nside  # this used to be user-defined.
        data = self.cmb_spectra
        lmax_cl = len(data[0]) + 1
        ell = np.arange(int(lmax_cl + 1))
        synlmax = min(synlmax, ell[-1])

        # Reading input spectra in CAMB format. CAMB outputs l(l+1)/2pi hence the corrections.
        cl_tebp_arr = np.zeros([10, lmax_cl + 1])
        cl_tebp_arr[0, 2:] = 2 * np.pi * data[1] / (ell[2:] * (ell[2:] + 1))  # TT
        cl_tebp_arr[1, 2:] = 2 * np.pi * data[2] / (ell[2:] * (ell[2:] + 1))  # EE
        cl_tebp_arr[2, 2:] = 2 * np.pi * data[3] / (ell[2:] * (ell[2:] + 1))  # BB
        cl_tebp_arr[4, 2:] = 2 * np.pi * data[4] / (ell[2:] * (ell[2:] + 1))  # TE
        cl_tebp_arr[5, :] = np.zeros(lmax_cl + 1)  # EB
        cl_tebp_arr[7, :] = np.zeros(lmax_cl + 1)  # TB

        if self.apply_delens:
            cl_tebp_arr[3, 2:] = (
                2
                * np.pi
                * data[5]
                * self.delensing_ells[1]
                / (ell[2:] * (ell[2:] + 1)) ** 2
            )  # PP
            cl_tebp_arr[6, :] = np.zeros(lmax_cl + 1)  # BP
            cl_tebp_arr[8, 2:] = (
                2
                * np.pi
                * data[7]
                * np.sqrt(self.delensing_ells[1])
                / (ell[2:] * (ell[2:] + 1)) ** 1.5
            )  # EP
            cl_tebp_arr[9, 2:] = (
                2
                * np.pi
                * data[6]
                * np.sqrt(self.delensing_ells[1])
                / (ell[2:] * (ell[2:] + 1)) ** 1.5
            )  # TP
        else:
            cl_tebp_arr[3, 2:] = (
                2 * np.pi * data[5] / (ell[2:] * (ell[2:] + 1)) ** 2
            )  # PP
            cl_tebp_arr[6, :] = np.zeros(lmax_cl + 1)  # BP
            cl_tebp_arr[8, 2:] = (
                2 * np.pi * data[7] / (ell[2:] * (ell[2:] + 1)) ** 1.5
            )  # EP
            cl_tebp_arr[9, 2:] = (
                2 * np.pi * data[6] / (ell[2:] * (ell[2:] + 1)) ** 1.5
            )  # TP

        # Coordinates of healpix pixel centers
        ipos = np.array(hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside))))

        # Simulate a CMB and lensing field
        cmb, aphi = simulate_tebp_correlated(
            cl_tebp_arr, self.nside, synlmax, self.cmb_seed
        )

        if cmb.ndim == 1:
            cmb = np.reshape(cmb, [1, cmb.size])

        # Compute the offset positions
        phi, phi_dtheta, phi_dphi = hp.alm2map_der1(aphi, self.nside, lmax=synlmax)

        del aphi

        opos, rot = offset_pos(
            ipos, phi_dtheta, phi_dphi, pol=True, geodesic=False
        )  # geodesic used to be used defined.
        del phi, phi_dtheta, phi_dphi

        # Interpolate maps one at a time
        maps = []
        for comp in cmb:
            for m in taylor_interpol_iter(
                comp, opos, 3, verbose=False, lmax=None
            ):  # lmax here needs to be fixed. order of taylor expansion is fixed to 3.
                pass
            maps.append(m)
        del opos, cmb
        return np.array(apply_rotation(maps, rot))
