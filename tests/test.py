import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import pysm3

if __name__ == "__main__":
    nside = 16
    bpass = [(np.linspace(140.0, 170.0, 10), np.ones(10))]
    freqs = np.linspace(1, 500, 100.0)

    sky = pysm3.Sky(nside, preset_strings=["d6"])
    sky.correlation_length = 1.0
    sky_d1 = pysm3.Sky(nside, preset_strings=["d1"])
    # hp.mollview(sky.get_emission(150.)[0, 0], title="model d4", norm='log')
    # plt.show()

    signal = sky.get_emission(freqs)
    signal_d1 = sky_d1.get_emission(freqs)
    fig, ax = plt.subplots(1, 1)
    print(signal.shape)
    ax.loglog(freqs, signal[:, 1, 0])
    ax.set_yscale("linear")
    plt.show()

    # get either individual model, or sky with group of components
    mbb = pysm3.preset_models("d1", nside)
    sky = pysm3.Sky(nside, preset_strings=["d1"])

    # the three main functions to be used (haven't added noise yet)
    out_delta = sky.get_emission(bpass[0][0].mean())
    out_bpass = sky.apply_bandpass(bpass)
    out_smoothed = sky.apply_smoothing(out_bpass, [120.0])

    hp.mollview(out_smoothed[0, 0], norm="log", title="bpass smoothed")
    hp.mollview(out_delta[0, 0], norm="log", title="delta bpass")
    hp.mollview(out_bpass[0, 0], norm="log", title="tophat bpass")
    plt.show()

    # or the same functionality using the individual component
    out_delta = mbb.get_emission(bpass[0][0].mean())
    out_bpass = mbb.apply_bandpass(bpass)
    out_smoothed = mbb.apply_smoothing(out_bpass, [120.0])

    hp.mollview(out_smoothed[0, 0], norm="log", title="bpass smoothed")
    hp.mollview(out_delta[0, 0], norm="log", title="delta bpass")
    hp.mollview(out_bpass[0, 0], norm="log", title="tophat bpass")
    plt.show()
