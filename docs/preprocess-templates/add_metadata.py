from astropy.io import fits
import sys
ext_number = 1

with fits.open(sys.argv[1], mode="update") as hdul:
    hdul[ext_number].header["REF_FREQ"] = "353 GHz"
    hdul[ext_number].header["COORDSYS"] = (
        "G",
        "Ecliptic, Galactic or Celestial (equatorial)",
    )
    for column in range(len(hdul[ext_number].columns)):
        hdul[ext_number].header[f"TUNIT{column+1}"] = sys.argv[2]
    hdul.flush()
