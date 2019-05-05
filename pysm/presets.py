""" Submodule containing preset models. The full list of the models available
can be found in the `initialize_model` function.

Functions:
    preset_models
    read_map
    d1
"""
from pathlib import Path
from .models import *
from .sky import Sky
import toml
from .constants import DATAURL
from astropy.utils import data

with data.conf.set_temp("dataurl", DATAURL):
    PRESET_MODELS = toml.load(data.get_pkg_data_filename("data/presets.cfg"))


def remove_class_from_dict(d):
    """Return a copy of dictionary without the key "class" """
    return {k: d[k] for k in d.keys() if k != "class"}


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
    try:
        class_name = config["class"]
    except KeyError:  # multiple components
        components = []
        for each_config in config.values():
            class_name = each_config["class"]
            component_class = globals()[class_name]
            components.append(
                component_class(**remove_class_from_dict(each_config), nside=nside)
            )
        output_component = Sky(component_objects=components, nside=nside)
    else:
        component_class = globals()[class_name]
        output_component = component_class(
            **remove_class_from_dict(config), nside=nside
        )
    return output_component
