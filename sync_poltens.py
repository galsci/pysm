""" This script takes a look at small-scale modelling of synchrotron emission.

We implement a "Frolov Transform", in which Gaussian small scales are added 
to log-combinations of the Stokes vectors, and then mapped back to Stokes
vectors, introducing intrinsic non-Gaussianities.

NOTES
=====
Please make sure you have set an environment variables telling the script where
to download data, e.g.:

> export PYSM_DATA_DIR=/path/to/scratch/space

"""

import os
from pathlib import Path
import wget

from absl import app, flags, logging

import numpy as np
import healpy as hp
from scipy.optimize import curve_fit

import pymaster as nmt
import pysm3.units as u

import cosmoplotian.colormaps
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
cmap = mpl.cm.get_cmap("div yel grn")
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k"] + [cmap(i) for i in [0.2, 0.5, 0.8]]) 

FLAGS = flags.FLAGS

DATA_DIR = Path(os.environ["PYSM_DATA_DIR"])

MAPS = {
    'PolMask': "COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits",
    'IntMask': "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
    'KBand': "wmap_band_iqumap_r9_9yr_K_v5.fits",
    'Haslam': "haslam408_dsds_Remazeilles2014.fits",
}


def VerifyData():
    """ If data not already downloaded, get it from the URLs listed.
    """
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    for tag in ["IntMask", "PolMask"]:
        fname = MAPS[tag]
        url = f"https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/{fname}"
        target = DATA_DIR / fname
        if not target.exists():
            wget.download(str(url), str(target))

    for tag in ["KBand"]:
        fname = MAPS["KBand"]
        url =  f"https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/raw/{fname}"
        target = DATA_DIR / fname
        if not target.exists():
            wget.download(str(url), str(target))

    for tag in ["Haslam"]:
        fname = MAPS["Haslam"]
        url =  f"https://lambda.gsfc.nasa.gov/data/foregrounds/haslam_2014/{fname}"
        target = DATA_DIR / fname
        if not target.exists():
            wget.download(str(url), str(target))


def FrolovTransformForward(IQU):
    """ From Stokes to polarization tensor
    """
    P = np.sqrt(IQU[1] ** 2 + IQU[2] ** 2)
    log_ratio = np.log((IQU[0] + P) / (IQU[0] - P))
    forward = np.zeros_like(IQU)
    forward[0] = 0.5 * np.log(IQU[0] ** 2 - P ** 2)
    forward[1:] = 0.5 * IQU[1:] * log_ratio / P
    return forward

def FrolovTransformBackward(iqu):
    """ From polarization tensor to Stokes
    """
    backward = np.zeros_like(iqu)
    ei = np.exp(iqu[0])
    p = np.sqrt(iqu[1] ** 2 + iqu[2] ** 2)
    backward[0] = ei * np.cosh(p)
    backward[1:] = iqu[1:] / p * ei * np.sinh(p)
    return backward


def InspectGNILCMasks():
    """ Plot intensity, polarization, and combined masks.
    """
    int_mask = hp.read_map(str(DATA_DIR / MAPS["IntMask"]))
    pol_mask = hp.read_map(str(DATA_DIR / MAPS["PolMask"]))
    joint_mask = np.logical_and(int_mask, pol_mask)

    hp.mollview(int_mask, title=r"${\rm Int~Mask}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig(Path(MAPS["IntMask"]).with_suffix('.pdf'), bbox_inches='tight')
    
    hp.mollview(pol_mask, title=r"${\rm Pol~Mask}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig(Path(MAPS["PolMask"]).with_suffix('.pdf'), bbox_inches='tight')
    
    hp.mollview(joint_mask, title=r"${\rm Joint~Mask}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig('JointMask.pdf', bbox_inches='tight')

    return


def InspectSyncMaps():
    """ Plot Haslam and Kband maps.
    """
    int_map = hp.read_map(str(DATA_DIR / MAPS["Haslam"])) * 1e6 * (23. / 0.408) ** -3
    pol_map = hp.read_map(str(DATA_DIR / MAPS["KBand"]), field=(1, 2)) * 1e3
    hp.mollview(int_map, min=0, max=500, title=r"${\rm Haslsm,~408~MHz~desourced~destriped}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig(Path(MAPS["Haslam"]).with_suffix('.pdf'), bbox_inches='tight')
    
    hp.mollview(pol_map[0], min=-250, max=250, title=r"${\rm WMAP~K~Band,~Q}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig(Path(MAPS["KBand"]).with_suffix('.pdf'),  bbox_inches='tight')

    hp.mollview(np.sqrt(pol_map[0] ** 2 + pol_map[1] ** 2), min=0, max=500, title=r"${\rm WMAP~K~Band,~P}$", cmap="div yel grn")
    fig = plt.gcf()
    fig.savefig("wmap_kband_P.pdf",  bbox_inches='tight')
    return


def DoFrolovTransform():
    """ Do the Frolov transform.
    """
    IQU = np.zeros((3, hp.nside2npix(512)))
    IQU[0, :] = hp.read_map(str(DATA_DIR / MAPS["Haslam"])) * 1e6 * (23. / 0.408) ** -3 # convert units, and scale to 23 GHz
    IQU[1:, :] = hp.read_map(str(DATA_DIR / MAPS["KBand"]), field=(1, 2)) * 1e3 # convert units

    # 0.88 degrees Kband res
    # 56 arcmin beam size haslam

    #IQU = hp.smoothing(IQU, fwhm=1. * np.pi/180.)

    iqu = FrolovTransformForward(IQU)

    hp.mollview(iqu[0], min=-2, max=10, title=r"Frolov $i$")
    fig = plt.gcf()
    fig.savefig("transformed_i_naive.pdf")
    hp.mollview(iqu[1], min=-5, max=5, title=r"Frolov $q$")
    fig = plt.gcf()
    fig.savefig("transformed_q_naive.pdf")
    hp.mollview(iqu[2], min=-5, max=5, title=r"Frolov $u$")
    fig = plt.gcf()
    fig.savefig("transformed_u_naive.pdf")

    hp.write_map(str(DATA_DIR / "transformed_iqu_naive.fits"), iqu, overwrite=True)

    iqu = FrolovTransformForward(hp.smoothing(IQU, fwhm=np.pi/180.))

    hp.mollview(iqu[0], min=-2, max=10, title=r"${\rm Frolov}~i$")
    fig = plt.gcf()
    fig.savefig("transformed_i_2degfwhm.pdf")
    hp.mollview(iqu[1], min=-2, max=2, title=r"${\rm Frolov}~q$")
    fig = plt.gcf()
    fig.savefig("transformed_q_2degfwhm.pdf")
    hp.mollview(iqu[2], min=-2, max=2, title=r"${\rm Frolov}~u$")
    fig = plt.gcf()
    fig.savefig("transformed_u_2degfwhm.pdf")

    hp.write_map(str(DATA_DIR / "transformed_iqu_fwhm2deg.fits"), iqu, overwrite=True)

    IQU = hp.smoothing(IQU, fwhm=2 * np.pi/180.)
    P = np.sqrt(IQU[1] ** 2 + IQU[2] ** 2)
    idx = np.where(P > IQU[0])[0]
    rescaling = np.sqrt(0.99 * IQU[0] ** 2 / P ** 2)
    IQU[1:, idx] *= rescaling[None, idx]
    iqu = FrolovTransformForward(IQU)

    hp.mollview(iqu[0], min=-2, max=10, title=r"${\rm Frolov}~i$")
    fig = plt.gcf()
    fig.savefig("transformed_i_2degfwhm_corrected.pdf")
    hp.mollview(iqu[1], min=-2, max=2, title=r"${\rm Frolov}~q$")
    fig = plt.gcf()
    fig.savefig("transformed_q_2degfwhm_corrected.pdf")
    hp.mollview(iqu[2], min=-2, max=2, title=r"${\rm Frolov}~u$")
    fig = plt.gcf()
    fig.savefig("transformed_u_2degfwhm_corrected.pdf")

    hp.write_map(str(DATA_DIR / "transformed_iqu_fwhm2deg_corrected.fits"), iqu, overwrite=True)
    return 


def CalculatePowerspectra(nside=512, lmax=1000):
    """ Calculate the TT, EE, BB, powerspectra of the raw Haslam 408 MHz
    map, and raw WMAP Kband map.

    This requires scaling the Haslam map to 23 GHz for TT comparison with
    polarization. We assume a spatially constant spectral index of -3 for
    this scaling.

    We also multiply the input Haslam map by 1e6 to convert to uK, and we
    multiple the input WMAP Kband by 1e3 to convert from mK to uK.

    The masks are the Planck cosmology intensity and polarization masks.
    """
    int_mask = hp.read_map(str(DATA_DIR / MAPS["IntMask"]))
    pol_mask = hp.read_map(str(DATA_DIR / MAPS["PolMask"]))
    int_map = hp.read_map(str(DATA_DIR / MAPS["Haslam"])) * 1e6 # map in K, convert to muK
    int_map *= (23. / 0.408) **  -3 # scale to 23 GHz
    pol_map = hp.read_map(str(DATA_DIR / MAPS["KBand"]), field=(1, 2)) * 1e3 # map in mK, convert to muK
    
    int_mask = hp.ud_grade(int_mask, nside_out=nside)
    pol_mask = hp.ud_grade(pol_mask, nside_out=nside)

    # combine int and pol masks. This will be necessary as
    # frolov transform mixes stokes parameters, requiring common
    # analysis region.
    joint_mask = np.logical_and(int_mask, pol_mask)
    for is_Dell in [True, False]:
        binning = nmt.NmtBin(nside=nside, nlb=1, lmax=lmax, is_Dell=is_Dell)
        f2 = nmt.NmtField(joint_mask, pol_map)
        f0 = nmt.NmtField(joint_mask, [int_map])

        cl_22 = nmt.compute_full_master(f2, f2, binning)
        cl_00 = nmt.compute_full_master(f0, f0, binning)

        cl = np.concatenate([binning.get_effective_ells()[None, :], cl_00, cl_22])
        if is_Dell:
            fname = "haslam_kband_spectra_dl.fits"
        else:
            fname = "haslam_kband_spectra_cl.fits"
        hp.write_cl(str(DATA_DIR / fname), cl, overwrite=True)

    iqu = hp.read_map(str(DATA_DIR / "transformed_iqu_fwhm2deg_corrected.fits"), field=(0, 1, 2))
    for is_Dell in [True, False]:
        binning = nmt.NmtBin(nside=nside, nlb=1, lmax=lmax, is_Dell=is_Dell)
        f2 = nmt.NmtField(joint_mask, iqu[1:])
        f0 = nmt.NmtField(joint_mask, [iqu[0]])

        cl_22 = nmt.compute_full_master(f2, f2, binning)
        cl_00 = nmt.compute_full_master(f0, f0, binning)

        cl = np.concatenate([binning.get_effective_ells()[None, :], cl_00, cl_22])
        if is_Dell:
            fname = "frolov_2degfwhm_corr_dl.fits"
        else:
            fname = "frolov_2degfwhm_corr_cl.fits"
        hp.write_cl(str(DATA_DIR / fname), cl, overwrite=True)

    return 

def PlotPowerspectra():
    """ Plot the powerspectra of the Haslam and WMAP Kband maps, scaled
    to 23 GHz, and plotted in uK^2.
    """
    dl = hp.read_cl(str(DATA_DIR / "haslam_kband_spectra_dl.fits"))
    fig, ax = plt.subplots(1, 1)
    ax.loglog(dl[0], dl[1], label=r"${\rm TT,~Haslam}$")    
    ax.loglog(dl[0], dl[2], label=r"${\rm EE,~K~band}$")
    ax.loglog(dl[0], dl[5], label=r"${\rm BB,~K~band}$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell (\ell + 1) / 2 \pi~C_\ell~({\rm \mu K}^2)$")
    ax.legend(frameon=False)
    ax.tick_params(direction="inout", which="both")
    ax.set_title(r"${\rm Data~at~23~GHz}$")
    fig.savefig("spectra_TTEEBB.pdf")

    dl = hp.read_cl(str(DATA_DIR / "frolov_2degfwhm_corr_dl.fits"))
    fig, ax = plt.subplots(1, 1)
    ax.loglog(dl[0], dl[1], label=r"${\rm tt}$")    
    ax.loglog(dl[0], dl[2], label=r"${\rm ee}$")
    ax.loglog(dl[0], dl[5], label=r"${\rm bb}$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell (\ell + 1) / 2 \pi~C_\ell~({\rm \mu K}^2)$")
    ax.legend(frameon=False)
    ax.tick_params(direction="inout", which="both")
    ax.set_title(r"${\rm Frolov~transformed~spectra}$")
    fig.savefig("frolov_spectra_tteebb.pdf")
    return


def PerformFitStokes(lmin=10, lmax=36):
    """ Fit power law model to sycnhrotron spectrum over a determined
    range of ells.
    """
    dl = hp.read_cl(str(DATA_DIR / "haslam_kband_spectra_dl.fits"))

    fig, ax = plt.subplots(1, 1)
    for (idx, label) in [(1, r"${\rm TT,~\gamma=}$"), (2, r"${\rm EE,~\gamma=}$"), (5, r"${\rm BB,~\gamma=}$")]:
        pars, cov = curve_fit(PowerLaw, dl[0, lmin:lmax], dl[idx, lmin:lmax])
        l1, = ax.loglog(dl[0], dl[idx], alpha=0.5)
        ax.loglog(dl[0], PowerLaw(dl[0], *pars), color=l1.get_color(), label=label+f"{pars[1]:.02f}")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell (\ell + 1) / 2 \pi~C_\ell~({\rm \mu K}^2)$")
    ax.legend(frameon=False)
    ax.tick_params(direction="inout", which="both")
    ax.set_title(r"${\rm Data~at~23~GHz}$")
    fig.savefig("fitted.pdf")
    return


def PowerLaw(ells, amplitude, gamma):
    return amplitude * ells ** gamma


def main(argv):
    del argv
    VerifyData()
    if FLAGS.mode == "all":
        InspectGNILCMasks()
        InspectSyncMaps()
        CalculatePowerspectra()
        PlotPowerspectra()
        PerformFitStokes()
        DoFrolovTransform()
    if FLAGS.mode == "inspect":
        #InspectGNILCMasks()
        InspectSyncMaps()
    if FLAGS.mode == "powerspectra":
        CalculatePowerspectra()
    if FLAGS.mode == "plot":
        PlotPowerspectra()
    if FLAGS.mode == "fit":
        PerformFitStokes()
    if FLAGS.mode == "transform":
        DoFrolovTransform()
    return

if __name__ == '__main__':
    flags.DEFINE_enum("mode", "inspect", ["all", "inspect", "powerspectra", "plot", "fit", "transform"], "Which mode to run in.")
    app.run(main)