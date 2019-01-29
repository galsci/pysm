""" Submodule containing models for the CIB.

ObjectS:
    CIB
"""
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from ..template import Model, check_freq_input


class InterpolatedCIB(Model):
    """ Model for CIB relying on interpolation between frequency samples
    of hydrodynamical simulations.
    """

    def __init__(self, map_dir=None, info_file=None, nside=None, pixel_indices=None, mpi_comm=None):
        """ 
        Parameters
        ----------
        map_dir: Path object
            Path object to the directory containing the CIB data.
        info_file: Path object
            Path object to the file containing the metadata baout the saved
            CIB maps. This assumes that the file will be in .txt format
            containing two columns, first the frequency in GHz at which the
            map is defined, and second the corresponding filename for the
            fits file containing the IQU maps at that frequency.
        """
        super().__init__(nside, pixel_indices=pixel_indices,mpi_comm=mpi_comm)
        try:
            assert isinstance(map_dir, Path)
        except AssertionError:
            print("map_dir must be an instance of pathlib.Path")
            raise
        # read in the metadata
        (freqs, cib_map_paths) = self.read_metadata(map_dir, info_file)
        # read in the CIB maps for each frequency
        cib_maps = [self.read_map(path, field=(0, 1, 2)) for path in cib_map_paths]
        # do the interpolation using a linear interpolation method from
        # `scipy.interpolate.interp1d`. We choose to return 0 outside of
        # the frequency range for which the maps are provided, instead of
        # raising an error.
        self.interp_sed = interpolate_cib_maps(freqs, cib_maps)
        return

    def read_metadata(self, map_dir, info_file):
        """ 
        """
        freqs = np.loadtxt(info_file)
        map_fpaths = [
            map_dir / "cib_map_{:04d}.fits".format(i) for i, _ in enumerate(freqs)
        ]
        return (freqs, map_fpaths)

    def get_emission(self, freqs):
        """ 
        """
        freqs = check_freq_input(freqs)
        return self.interp_sed(freqs)


def interpolate_cib_maps(freqs, cib_map_arr):
    """ Function to take a set of CIB maps and compute a linear spline
    interpolation between them.

    Parameters
    ----------
    freqs: ndarray
        Array containing the frequencies at which the input CIB maps are
        defined.
    cib_map_arr: ndarray
        Array containing the CIB maps, a set of IQU maps for each frequency.

    Returns
    -------
    spline object
        This spline is a function of frequency, which returns an `ndarray`
        of shape (3, npix), corresponding to the CIB emission at that
        that frequency.
    """
    return interp1d(
        freqs,
        cib_map_arr,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(0., 0.),
    )
