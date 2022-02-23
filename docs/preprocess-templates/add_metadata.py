from astropy.io import fits
import sys

with fits.open(sys.argv[1], mode="update") as hdul:
    hdul[1].header["REF_FREQ"] = "353 GHz"
    hdul[1].header["COORDSYS"] = (
        "G",
        "Ecliptic, Galactic or Celestial (equatorial)",
    )
    for column in [1, 2, 3]:
        hdul[1].header[f"TUNIT{column}"] = sys.argv[2]
    hdul.flush()
