import healpy as hp
import numpy as np

from .. import units as u
from .. import utils
from .dust import blackbody_ratio
from .template import Model


class GaussianPolarizedForeground(Model):
    r"""Base class for Gaussian polarized foreground components.

    This model generates one Gaussian realization in harmonic space from power
    laws in :math:`D_\ell` and converts it to Q/U maps at a pivot frequency.
    Derived classes implement component-specific frequency scaling in
    thermodynamic units.
    """

    def __init__(
        self,
        nside,
        amplitude_ee,
        amplitude_bb,
        amplitude_tt,
        alpha_ee,
        alpha_bb,
        alpha_tt,
        freq_pivot,
        seed=None,
        lmax=None,
        max_nside=None,
        map_dist=None,
    ):
        if map_dist is not None:
            msg = "Gaussian foreground components currently do not support MPI map distribution"
            raise NotImplementedError(msg)

        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)

        self.has_polarization = True
        self.amplitude_ee = float(amplitude_ee)
        self.amplitude_bb = float(amplitude_bb)
        self.amplitude_tt = float(amplitude_tt)
        self.alpha_ee = float(alpha_ee)
        self.alpha_bb = float(alpha_bb)
        self.alpha_tt = float(alpha_tt)
        self.freq_pivot = u.Quantity(freq_pivot).to(u.GHz)
        self.seed = seed
        self.lmax = int(3 * nside - 1 if lmax is None else lmax)

        self.ells = np.arange(self.lmax + 1)
        self.cl_tt, self.cl_ee, self.cl_bb = self._build_power_spectra()
        self.I_ref, self.Q_ref, self.U_ref = self._draw_iqu_maps()

        self._pivot_kcmb_to_jysr = self._kcmb_to_jysr(self.freq_pivot.value)

    def _build_power_spectra(self):
        cl_tt = np.zeros_like(self.ells, dtype=np.float64)
        cl_ee = np.zeros_like(self.ells, dtype=np.float64)
        cl_bb = np.zeros_like(self.ells, dtype=np.float64)
        dell_factor = self.ells * (self.ells + 1) / (2 * np.pi)

        valid = self.ells >= 2
        cl_tt[valid] = (
            self.amplitude_tt * (self.ells[valid] / 80.0) ** self.alpha_tt / dell_factor[valid]
        )
        cl_ee[valid] = (
            self.amplitude_ee * (self.ells[valid] / 80.0) ** self.alpha_ee / dell_factor[valid]
        )
        cl_bb[valid] = (
            self.amplitude_bb * (self.ells[valid] / 80.0) ** self.alpha_bb / dell_factor[valid]
        )
        return cl_tt, cl_ee, cl_bb

    def _draw_iqu_maps(self):
        rng = np.random.default_rng(self.seed)
        alm_tt = draw_synalm_from_cl(self.cl_tt, self.lmax, rng)
        alm_ee = draw_synalm_from_cl(self.cl_ee, self.lmax, rng)
        alm_bb = draw_synalm_from_cl(self.cl_bb, self.lmax, rng)
        i_map = hp.alm2map(alm_tt, self.nside, lmax=self.lmax)

        q_map, u_map = hp.alm2map_spin([alm_ee, alm_bb], self.nside, 2, lmax=self.lmax)
        return i_map, q_map, u_map

    def _kcmb_to_jysr(self, freq_ghz):
        with u.set_enabled_equivalencies(u.cmb_equivalencies(freq_ghz * u.GHz)):
            return (1.0 * u.K_CMB).to_value(u.Jy / u.sr)

    def scaling_to_pivot_in_cmb(self, freq_ghz):
        """Return frequency scaling relative to pivot in thermodynamic units."""
        raise NotImplementedError

    def get_dell_theory(self, freq_ghz, ells=None):
        """Return theoretical D_ell (TT, EE, BB) at a given frequency in uK_RJ^2."""
        if ells is None:
            ells = self.ells
        ells = np.asarray(ells)
        scaling = self.scaling_to_pivot_in_cmb(freq_ghz)
        with u.set_enabled_equivalencies(u.cmb_equivalencies(freq_ghz * u.GHz)):
            cmb_to_rj = (1.0 * u.uK_CMB).to_value(u.uK_RJ)
        amp_scale = (scaling * cmb_to_rj) ** 2

        dell_tt = np.zeros_like(ells, dtype=np.float64)
        dell_ee = np.zeros_like(ells, dtype=np.float64)
        dell_bb = np.zeros_like(ells, dtype=np.float64)
        valid = ells >= 2

        dell_tt[valid] = self.amplitude_tt * (ells[valid] / 80.0) ** self.alpha_tt * amp_scale
        dell_ee[valid] = self.amplitude_ee * (ells[valid] / 80.0) ** self.alpha_ee * amp_scale
        dell_bb[valid] = self.amplitude_bb * (ells[valid] / 80.0) ** self.alpha_bb * amp_scale
        return {"TT": dell_tt, "EE": dell_ee, "BB": dell_bb}

    @u.quantity_input
    def get_emission(self, freqs: u.Quantity[u.GHz], weights=None) -> u.Quantity[u.uK_RJ]:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)

        output = np.zeros((3, self.I_ref.size), dtype=np.float64)
        temp = np.zeros_like(output) if len(freqs) > 1 else output

        for i, (freq, _weight) in enumerate(zip(freqs, weights)):
            scaling = self.scaling_to_pivot_in_cmb(freq)

            temp[0] = self.I_ref * scaling
            temp[1] = self.Q_ref * scaling
            temp[2] = self.U_ref * scaling

            with u.set_enabled_equivalencies(u.cmb_equivalencies(freq * u.GHz)):
                cmb_to_rj = (1.0 * u.uK_CMB).to_value(u.uK_RJ)
            temp *= cmb_to_rj

            if len(freqs) > 1:
                utils.trapz_step_inplace(freqs, weights, i, temp, output)

        return output << u.uK_RJ


class GaussianDust(GaussianPolarizedForeground):
    r"""Gaussian polarized dust foreground model.

    Defaults match the foreground parameters used in the Simons Observatory
    Gaussian-foreground simulations from Wolz et al. (2024) for EE/BB.

    TT simulation recipe:

    1. Define a power-law TT spectrum in :math:`D_\ell`:
       :math:`D_\ell^{TT} = A_{TT} (\ell / 80)^{\alpha_{TT}}`.
    2. Convert to :math:`C_\ell` with
       :math:`C_\ell = D_\ell \, 2\pi / [\ell(\ell+1)]`.
    3. Draw Gaussian :math:`a_{\ell m}` from this :math:`C_\ell` and synthesize
       the temperature map with ``healpy``.

    Default TT values are :math:`A_{TT}=5600` and :math:`\alpha_{TT}=-0.8`
    (equivalently :math:`C_\ell^{TT} \propto \ell^{-2.8}`), chosen to be
    consistent with diffuse dust/cirrus intensity angular power spectrum
    measurements, e.g.
    Miville-Deschenes et al. (2007): https://arxiv.org/abs/0708.4414
    """

    def __init__(
        self,
        nside,
        amplitude_ee=56.0,
        amplitude_bb=28.0,
        amplitude_tt=5600.0,
        alpha_ee=-0.32,
        alpha_bb=-0.16,
        alpha_tt=-0.8,
        freq_pivot="353 GHz",
        beta=1.54,
        temperature=20.0,
        seed=None,
        lmax=None,
        max_nside=None,
        map_dist=None,
    ):
        self.beta = float(beta)
        self.temperature = u.Quantity(temperature, u.K).to_value(u.K)
        super().__init__(
            nside=nside,
            amplitude_ee=amplitude_ee,
            amplitude_bb=amplitude_bb,
            amplitude_tt=amplitude_tt,
            alpha_ee=alpha_ee,
            alpha_bb=alpha_bb,
            alpha_tt=alpha_tt,
            freq_pivot=freq_pivot,
            seed=seed,
            lmax=lmax,
            max_nside=max_nside,
            map_dist=map_dist,
        )

    def scaling_to_pivot_in_cmb(self, freq_ghz):
        blackbody = blackbody_ratio(freq_ghz, self.freq_pivot.value, self.temperature)
        emissivity = (freq_ghz / self.freq_pivot.value) ** (self.beta - 2.0)
        unit_ratio = self._pivot_kcmb_to_jysr / self._kcmb_to_jysr(freq_ghz)
        return blackbody * emissivity * unit_ratio


class GaussianSynchrotron(GaussianPolarizedForeground):
    r"""Gaussian polarized synchrotron foreground model.

    Defaults match the foreground parameters used in the Simons Observatory
    Gaussian-foreground simulations from Wolz et al. (2024) for EE/BB.

    TT simulation recipe:

    1. Define a power-law TT spectrum in :math:`D_\ell`:
       :math:`D_\ell^{TT} = A_{TT} (\ell / 80)^{\alpha_{TT}}`.
    2. Convert to :math:`C_\ell` with
       :math:`C_\ell = D_\ell \, 2\pi / [\ell(\ell+1)]`.
    3. Draw Gaussian :math:`a_{\ell m}` from this :math:`C_\ell` and synthesize
       the temperature map with ``healpy``.

    Default TT values are :math:`A_{TT}=100` and :math:`\alpha_{TT}=-0.8`
    (equivalently :math:`C_\ell^{TT} \propto \ell^{-2.8}`), which lies inside
    the diffuse synchrotron intensity angular power spectrum range (roughly
    :math:`C_\ell \propto \ell^{-2.6}` to :math:`\ell^{-3.0}`), e.g.
    La Porta et al. (2008): https://arxiv.org/abs/0804.4587
    """

    def __init__(
        self,
        nside,
        amplitude_ee=9.0,
        amplitude_bb=1.6,
        amplitude_tt=100.0,
        alpha_ee=-0.7,
        alpha_bb=-0.93,
        alpha_tt=-0.8,
        freq_pivot="23 GHz",
        beta=-3.0,
        seed=None,
        lmax=None,
        max_nside=None,
        map_dist=None,
    ):
        self.beta = float(beta)
        super().__init__(
            nside=nside,
            amplitude_ee=amplitude_ee,
            amplitude_bb=amplitude_bb,
            amplitude_tt=amplitude_tt,
            alpha_ee=alpha_ee,
            alpha_bb=alpha_bb,
            alpha_tt=alpha_tt,
            freq_pivot=freq_pivot,
            seed=seed,
            lmax=lmax,
            max_nside=max_nside,
            map_dist=map_dist,
        )

    def scaling_to_pivot_in_cmb(self, freq_ghz):
        synch_scaling = (freq_ghz / self.freq_pivot.value) ** self.beta
        unit_ratio = self._pivot_kcmb_to_jysr / self._kcmb_to_jysr(freq_ghz)
        return synch_scaling * unit_ratio


def draw_synalm_from_cl(cl, lmax, rng):
    """Draw scalar alm coefficients from C_ell using a NumPy Generator."""
    size = hp.Alm.getsize(lmax)
    ell, m = hp.Alm.getlm(lmax, np.arange(size))
    sigma = np.sqrt(cl[ell])

    alm = np.zeros(size, dtype=np.complex128)
    m0 = m == 0
    alm[m0] = rng.normal(scale=sigma[m0])

    mpos = ~m0
    sigma_half = sigma[mpos] / np.sqrt(2.0)
    alm[mpos] = rng.normal(scale=sigma_half) + 1j * rng.normal(scale=sigma_half)
    return alm
