import numpy as np


def car_aperture_photometry(
    thumbs, aperture_radius, annulus_width=None, modrmap=None, pixsizemap=None
):
    """
    Flux from aperture photometry.

    from https://github.com/msyriac/orphics/blob/master/orphics/maps.py

    Parameters
    ----------
    thumb : ndmap
        An (...,Ny,Nx) ndmap (i.e. a pixell enmap) containing the thumbnails.
    aperture_radius : float
        Aperture inner radius in radians
    annulus_width : float
        Annulus width for mean subtraction in radians.
        Defaults to sqrt(2)-1 times the aperture inner radius.
    modrmap : ndmap, optional
        An (Ny,Nx) ndmap containing distances of each pixel from the center in radians.
    modrmap : ndmap, optional
        An (Ny,Nx) ndmap containing pixel areas in steradians.

    Returns
    -------
    flux : ndarray
        (...,) array of aperture photometry fluxes.

    """
    if modrmap is None:
        modrmap = thumbs.modrmap()
    if annulus_width is None:
        annulus_width = (np.sqrt(2.0) - 1.0) * aperture_radius
    # Get the mean background level from the annulus
    mean = thumbs[
        ...,
        np.logical_and(
            modrmap > aperture_radius, modrmap < (aperture_radius + annulus_width)
        ),
    ].mean()
    if pixsizemap is None:
        pixsizemap = thumbs.pixsizemap()
    # Subtract the mean, multiply by pixel areas and sum
    return (((thumbs - mean) * pixsizemap)[..., modrmap <= aperture_radius]).sum(
        axis=-1
    )
