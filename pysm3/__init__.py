# flake8: noqa
from ._astropy_init import *  # noqa
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.filterwarnings("ignore", message="may indicate binary incompatibility")

try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from .models import *
from .sky import Sky, get_pysm_emission
from . import units
from .distribution import MapDistribution
from .mpi import mpi_smoothing
from .utils import normalize_weights, bandpass_unit_conversion, check_freq_input
