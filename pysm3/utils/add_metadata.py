from astropy.io import fits


def add_metadata(filenames, field=1, coord=None, unit=None, **kwargs):
    """Add metadata to an existing FITS file

    Parameters
    ----------
    filenames : list
        list of paths to FITS files
    field : int
        FITS extension index
    coord : str
        Reference frame, E, G or C (equatorial)
    unit : str or list
        Unit for each column or same unit for all columns
    kwargs : keywords
        Other keywords are added to the header as key,value
        pairs, key is transformed to uppercase
    """

    for filename in filenames:
        with fits.open(filename, mode="update") as hdul:
            hdu = hdul[field]
            if coord is not None:
                assert coord in "EGC", "Invalid reference frame"
                hdu.header["COORDSYS"] = (
                    coord,
                    "Ecliptic, Galactic or Celestial (equatorial)",
                )
            if unit is not None:
                num_columns = len(hdu.columns)
                for column in range(num_columns):
                    hdu.header[f"TUNIT{column+1}"] = (
                        str(unit) if not isinstance(unit, list) else str(unit[column])
                    )
            for k, v in kwargs.items():
                hdu.header[k.upper()] = v
            hdul.flush()
