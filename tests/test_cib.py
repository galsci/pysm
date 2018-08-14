import sys
sys.path.append('/home/ben/Projects/PySM/PySM_public')
import unittest
import numpy as np
import healpy as hp
from pathlib import Path
from pysm.component_models.extragalactic.cib import InterpolatedCIB

class TestInterpolatedCIB(unittest.TestCase):
    def setUp(self):
        """ Create some fake data that corresponds to a linear SED, which
        the linear interpolation should capture perfectly. Calculate the
        interpolation here, and through the InterpolatedCIB object.
        """
        self.nside = 16
        nsamps = 100
        self.npix = hp.nside2npix(self.nside)
        # define the freuqencies of the anchors.
        self.freqs_sample = np.linspace(10, 1000, nsamps)
        # define the test frequencies shifted from the sampled ones.
        self.freqs_test = np.linspace(10.5, 999.5, nsamps - 1)
        # create some fake test data
        ref_amp_map = np.random.randn(3, self.npix)
        freq_0 = 150.
        self.samples = linear_sed(ref_amp_map, self.freqs_sample, freq_0)
        self.test_samples = linear_sed(ref_amp_map, self.freqs_test, freq_0)
        # save some of this data to a temporary directory.
        self.temp_dir = Path(__file__).absolute().parent / 'temp_dir'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.save_info_file()
        self.save_maps()
        return

    def tearDown(self):
        # remove all temporary files
        for i, _ in enumerate(self.freqs_sample):
            self.temp_cib_map_name(i).unlink()
        self.info_file_path().unlink()
        # remove temporary directory
        self.temp_dir.rmdir()
        
        return

    def test_emission(self):
        cib = InterpolatedCIB(map_dir=self.temp_dir,
                              info_file=self.info_file_path(),
                              nside=self.nside)
        test_interp = cib.get_emission(self.freqs_test)
        np.assert_almost_equal(test_interp, self.test_samples)
        return

    def save_maps(self):
        for i, _ in enumerate(self.freqs_sample):
            fpath = str(self.temp_cib_map_name(i))
            hp.write_map(fpath, self.samples[i], overwrite=True)
        return None

    def save_info_file(self):
        path = self.info_file_path()
        np.savetxt(path, self.freqs_sample.T)
        return None
    
    def info_file_path(self):
        return self.temp_dir / 'info_file.txt'
        
    def temp_cib_map_name(self, i):
        print(i)
        return self.temp_dir / 'cib_map_{:04d}.fits'.format(i)

def linear_sed(amp_maps, freqs, freq_0):
    """ Function to make mock CIB maps using a linear SED. This is not 
    physically motivated, but used as a test of the implementation of 
    linear interpolation.

    Parameters
    ----------
    amp_maps: ndarray
        Array containing some random numbers, of shape (3, npix).
    freqs: ndarray
        Array containing the frequencies to which to scale `amp_map`.
    freq_0: float
        Frequency at which `amp_maps` is defined.

    Return
    ------
    ndarray
        (len(freqs), 3, npix) array containing the scale CIB maps.
    """
    return amp_maps[None, :] * (freqs / freq_0)[:, None, None]

if __name__ == '__main__':
    unittest.main()
