import healpy as hp
import numpy as np

from .. import units as u
from .. import utils
from .template import Model


def build_lines_dict(lines, maps):
    """Build a dictionary for lines and maps

    Takes a list of tags (strings) and a map or set of maps
    and returns a dictionary where each tag is associated with
    a map
    """
    return dict(zip(lines, np.atleast_2d(maps)))


class COLines(Model):
    def __init__(
        self,
        nside,
        max_nside=None,
        has_polarization=True,
        lines=["10", "21", "32"],
        include_high_galactic_latitude_clouds=False,
        polarization_fraction=0.001,
        theta_high_galactic_latitude_deg=20.0,
        random_seed=1234567,
        verbose=False,
        run_mcmole3d=False,
        map_dist=None,
    ):
        """Class defining attributes for CO line emission.
        CO templates are extracted from Type 1 CO Planck maps.
        See further details in:
        https://www.aanda.org/articles/aa/abs/2014/11/aa21553-13/aa21553-13.html

        Parameters
        ----------
        nside : int
            HEALPix NSIDE of the output maps
        has_polarization : bool
            whether or not to simulate also polarization maps
        lines : list of strings
            CO rotational transitions to consider.
            Accepted values : 10, 21, 32
        polarization_fraction: float
            polarisation fraction for polarised CO emission.
        include_high_galactic_latitude_clouds: bool
            If True it includes a simulation from MCMole3D to include
            high Galactic Latitude clouds.
            (See more details at http://giuspugl.github.io/mcmole/index.html)
        run_mcmole3d: bool
            If True it simulates  HGL cluds by running MCMole3D, otherwise it coadds
            a map of HGL emission.
        random_seed: int
            set random seed for mcmole3d simulations.
        theta_high_galactic_latitude_deg : float
            Angle in degree  to identify High Galactic Latitude clouds
            (i.e. clouds whose latitude b is `|b|> theta_high_galactic_latitude_deg`).
        map_dist : mpi4py communicator
            Read inputs across a MPI communicator, see pysm.read_map
        """

        self.lines = lines
        self.line_index = {"10": 0, "21": 1, "32": 2}
        self.line_frequency = {
            "10": 115.271 * u.GHz,
            "21": 230.538 * u.GHz,
            "32": 345.796 * u.GHz,
        }
        self.nside = nside

        self.template_nside = 512 if self.nside <= 512 else 2048

        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)

        self.remote_data = utils.RemoteData()

        self.planck_templatemap_filename = (
            f"co/HFI_CompMap_CO-Type1_{self.template_nside}_R2.00_ring.fits"
        )
        self.planck_templatemap = build_lines_dict(
            self.lines,
            hp.ud_grade(
                map_in=self.read_map(
                    self.remote_data.get(self.planck_templatemap_filename),
                    field=[self.line_index[line] for line in self.lines],
                    unit=u.K_CMB,
                ),
                nside_out=self.nside,
            )
            << u.K_CMB,
        )

        self.include_high_galactic_latitude_clouds = (
            include_high_galactic_latitude_clouds
        )
        self.has_polarization = has_polarization
        if self.has_polarization:
            self.polangle = self.read_map(
                self.remote_data.get(f"co/psimap_dust90_{self.template_nside}.fits")
            ).value
            self.depolmap = self.read_map(
                self.remote_data.get(f"co/gmap_dust90_{self.template_nside}.fits")
            ).value
        self.polarization_fraction = polarization_fraction
        self.theta_high_galactic_latitude_deg = theta_high_galactic_latitude_deg
        self.random_seed = random_seed
        self.run_mcmole3d = run_mcmole3d

        if include_high_galactic_latitude_clouds and not run_mcmole3d:
            # Dictionary where keys are "10", "21" and values
            self.mapclouds = build_lines_dict(
                self.lines,
                self.read_map(
                    self.remote_data.get(f"co/mcmoleCO_HGL_{self.template_nside}.fits"),
                    field=[self.line_index[line] for line in self.lines],
                    unit=u.K_CMB,
                ),
            )

        self.verbose = verbose

    @u.quantity_input
    def get_emission(
        self, freqs: u.Quantity[u.GHz], weights=None
    ) -> u.Quantity[u.uK_RJ]:
        freqs = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(freqs, weights)
        out = np.zeros((3, hp.nside2npix(self.nside)), dtype=np.float64)
        for line in self.lines:
            line_freq = self.line_frequency[line].to_value(u.GHz)
            if line_freq >= freqs[0] and line_freq <= freqs[-1]:
                weight = np.interp(line_freq, freqs, weights)
                convert_to_uK_RJ = (1 * u.K_CMB).to_value(
                    u.uK_RJ,
                    equivalencies=u.cmb_equivalencies(line_freq * u.GHz),
                )
                I_map = self.planck_templatemap[line].copy()
                if self.include_high_galactic_latitude_clouds:
                    I_map += self.simulate_high_galactic_latitude_CO(line)

                if self.has_polarization:
                    out[1:] += (
                        self.simulate_polarized_emission(I_map).value
                        * convert_to_uK_RJ
                        * weight
                    )
                out[0] += I_map.value * convert_to_uK_RJ * weight

        return out << u.uK_RJ

    def simulate_polarized_emission(self, I_map):
        """Add polarized emission by means of:
        * an overall constant polarization fraction,
        * a depolarization map to mimick the line of sight depolarization effect
        at low Galactic latitudes
        * a polarization angle map coming from a dust template (we exploit the
        observed correlation between polarized dust and molecular emission in star forming regions).
        """

        cospolangle = np.cos(2.0 * self.polangle)
        sinpolangle = np.sin(2.0 * self.polangle)

        P_map = self.polarization_fraction * self.depolmap * I_map
        return P_map * np.array([cospolangle, sinpolangle])

    def simulate_high_galactic_latitude_CO(self, line):
        """
        Coadd High Galactic Latitude CO emission, simulated with  MCMole3D.
        """
        if self.run_mcmole3d:
            import mcmole3d as cl

            # params to MCMole
            N = 40000
            L_0 = 20.4  # pc
            L_min = 0.3
            L_max = 60.0
            R_ring = 5.8
            sigma_ring = 2.7  # kpc
            R_bulge = 3.0
            R_z = 10  # kpc
            z_0 = 0.1
            Em_0 = 240.0
            R_em = 6.6
            model = "LogSpiral"

            nside = self.nside
            Itot_o, _ = cl.integrate_intensity_map(
                self.planck_templatemap[line],
                hp.get_nside(self.planck_templatemap[line]),
                planck_map=True,
            )
            Pop = cl.Cloud_Population(N, model, randseed=self.random_seed)

            Pop.set_parameters(
                radial_distr=[R_ring, sigma_ring, R_bulge],
                typical_size=L_0,
                size_range=[L_min, L_max],
                thickness_distr=[z_0, R_z],
                emissivity=[Em_0, R_em],
            )
            Pop()

            if self.verbose:
                Pop.print_parameters()
            # project into  Healpix maps
            mapclouds = cl.do_healpy_map(
                Pop,
                nside,
                highgalcut=np.deg2rad(90.0 - self.theta_high_galactic_latitude_deg),
                apodization="gaussian",
                verbose=self.verbose,
            )
            Itot_m, _ = cl.integrate_intensity_map(mapclouds, nside)
            # convert simulated map into the units of the Planck one
            rescaling_factor = Itot_m / Itot_o
            mapclouds /= rescaling_factor
            hglmask = np.zeros_like(mapclouds)
            # Apply mask to low galactic latitudes
            listhgl = hp.query_strip(
                nside,
                np.deg2rad(90.0 + self.theta_high_galactic_latitude_deg),
                np.deg2rad(90 - self.theta_high_galactic_latitude_deg),
            )
            hglmask[listhgl] = 1.0
            rmsplanck = self.planck_templatemap[line][listhgl].std()
            rmssim = mapclouds[listhgl].std()
            belowplanck = 1.0 if rmssim == 0.0 else rmssim / rmsplanck

            return mapclouds * hglmask / belowplanck
        return self.mapclouds[line]
