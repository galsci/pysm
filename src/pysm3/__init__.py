# flake8: noqa
import warnings

warnings.filterwarnings("ignore", message="may indicate binary incompatibility")

from ._version import __version__, __version_tuple__

from .models import *
from .sky import Sky, get_pysm_emission
from . import units
from .distribution import MapDistribution
from .mpi import mpi_smoothing
from .utils import (
    normalize_weights,
    bandpass_unit_conversion,
    check_freq_input,
    set_verbosity,
    apply_smoothing_and_coord_transform,
    map2alm,
)
# Import functions that don't require scikit-learn
from .bandpass_sampler import (
    bandpass_distribution_function,
    compute_moments,
)

# Conditionally import functions that require scikit-learn
try:
    from .bandpass_sampler import (
        search_optimal_kernel_bandwidth,
        bandpass_kresampling,
        resample_bandpass,
    )
except ImportError:
    # Functions will raise ImportError when called if scikit-learn is not available
    # This allows the module to load but provides clear error messages
    pass
