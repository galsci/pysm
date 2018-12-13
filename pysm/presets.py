""" Submodule containing preset models. The full list of the models available
can be found in the `initialize_model` function.

Functions:
    preset_models
    read_map
    d1
"""
from pathlib import Path
from .component_models import ModifiedBlackBody, DecorrelatedModifiedBlackBody, SynchrotronPowerLaw
from .sky import Sky
from configobj import ConfigObj
from astropy.utils import data
data.conf.dataurl = "https://healpy.github.io/pysm-data/"

PRESET_MODELS = ConfigObj(data.get_pkg_data_filename("data/presets.cfg"))

def preset_models(model_string, nside):
    """ Function to take a given model string, and nside, and construct
    the corresponding model object.

    This function contains useful presets corresponding to commonly used
    foreground models such as single component modified black body, and
    power law synchrotorn radiation.

    Parameters
    ----------
    model_string: str
        String identifying which model to set up.
    nside: int
        Resolution parameter indicating resolution at which to define models.
        Specified the resolution at which template maps will be read in.

    Notes
    -----
    The full list of available models in this function is:
    """
    config = PRESET_MODELS[model_string]
    class_name = config.pop("class")
    component_class = globals()[class_name]
    return component_class(**config, nside=nside)

