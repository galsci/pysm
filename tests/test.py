import sys
from pathlib import Path
sys.path.append('/home/ben/Projects/PySM/pysmv3')
import pysm
import unittest
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nside = 256
    bpass = [
        (np.linspace(140., 170., 10), np.ones(10))
    ]

    # get either individual model, or sky with group of components
    mbb = pysm.preset_models('d1', nside)
    sky = pysm.Sky(nside, preset_strings=['d1'])

    # the three main functions to be used (haven't added noise yet)
    out_delta = sky.get_emission(bpass[0][0].mean())
    out_bpass = sky.apply_bandpass(bpass)
    out_smoothed = sky.apply_smoothing(out_bpass, [120.])

    hp.mollview(out_smoothed[0, 0], norm='log', title='bpass smoothed')
    hp.mollview(out_delta[0, 0], norm='log', title='delta bpass')
    hp.mollview(out_bpass[0, 0], norm='log', title='tophat bpass')
    plt.show()

    # or the same functionality using the individual component
    out_delta = mbb.get_emission(bpass[0][0].mean())
    out_bpass = mbb.apply_bandpass(bpass)
    out_smoothed = mbb.apply_smoothing(out_bpass, [120.])

    hp.mollview(out_smoothed[0, 0], norm='log', title='bpass smoothed')
    hp.mollview(out_delta[0, 0], norm='log', title='delta bpass')
    hp.mollview(out_bpass[0, 0], norm='log', title='tophat bpass')
    plt.show()
