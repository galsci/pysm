""" This submoduls contains most of the user-facing code, as the objects
`Sky` and `Instrument` can be used for most of the functionality of the
code, without delving into the difference `Model` objects.

Objects:
    Sky
"""

import inspect

import toml

from . import units as u
from .models import *
from .models import Model
from .utils import bandpass_unit_conversion


def remove_class_from_dict(d):
    """Return a copy of dictionary without the key "class" """
    return {k: d[k] for k in d if k != "class"}


def _component_init(class_obj, config_kwargs, nside, map_dist, smoothing_angle):
    """Instantiate a component class, forwarding ``smoothing_angle`` only to the
    classes that accept it (template models that support presmoothing)."""
    component_kwargs = remove_class_from_dict(config_kwargs)
    if (
        smoothing_angle is not None
        and "smoothing_angle" in inspect.signature(class_obj).parameters
    ):
        component_kwargs["smoothing_angle"] = smoothing_angle
    return class_obj(**component_kwargs, nside=nside, map_dist=map_dist)


def create_components_from_config(config, nside, map_dist=None, smoothing_angle=None):
    output_components = []
    if "class" in config:
        class_name = config["class"]
        component_class = globals()[class_name]
        output_component = _component_init(
            component_class,
            config,
            nside,
            map_dist,
            smoothing_angle,
        )
        output_components.append(output_component)
        return output_components

    for model_name, model_config in config.items():
        try:
            class_name = model_config["class"]
        except KeyError:  # multiple components
            partial_components = []
            for each_config in model_config.values():
                class_name = each_config["class"]
                component_class = globals()[class_name]
                partial_components.append(
                    _component_init(
                        component_class,
                        each_config,
                        nside,
                        map_dist,
                        smoothing_angle,
                    )
                )
            output_component = Sky(
                component_objects=partial_components,
                nside=nside,
                map_dist=map_dist,
                smoothing_angle=smoothing_angle,
            )
        else:
            component_class = globals()[class_name]
            output_component = _component_init(
                component_class,
                model_config,
                nside,
                map_dist,
                smoothing_angle,
            )
        output_components.append(output_component)
    return output_components


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


PRESET_MODELS = toml.loads(
    pkg_resources.files("pysm3").joinpath("data", "presets.cfg").read_text()
)


def get_pysm_emission(preset_string, nside):
    """Get one of PySM preset emissions

    Parameters
    ----------
    preset_string : str
        PySM 2 or 3 model string, for example "d10"
    nside : int
        Requested Nside

    Returns
    -------
    comp : subclass of pysm3.Model
        PySM Model object
    """
    component_config = {preset_string: PRESET_MODELS[preset_string]}
    return create_components_from_config(component_config, nside=nside)


class Sky(Model):
    """Sky is the main interface to PySM

    Please read the 'Best practices for model execution' section in the
    documentation homepage before running PySM 3 models.

    It accepts the configuration of the desired components in 3 different
    ways: `preset_strings`, `component_config` or `component_objects`,
    see details below.
    Once a Sky object is created, all the sky components are initialized,
    i.e. loading the input templates.
    Then bandpass-integrated maps can be computed calling the
    `get_emission` method.
    Check the :func:`~pysm.apply_smoothing_and_coord_transform` function
    for applying a beam and transform coordinates to the map arrays
    from `get_emission`.
    See the tutorials section of the documentation for examples.

    Attributes
    ----------
    components: list(pysm.Model object)
        List of `pysm.Model` objects.
    """

    def __init__(
        self,
        nside=None,
        max_nside=None,
        preset_strings=None,
        component_config=None,
        component_objects=None,
        output_unit=u.uK_RJ,
        smoothing_angle=None,
        map_dist=None,
    ):
        """Initialize Sky

        Parameters
        ----------
        nside : int
            Requested output NSIDE, inputs will be degraded
            using :func:`healpy.ud_grade`
        max_nside: int
            Keeps track of the the maximum Nside this model is available at
            by default 512 like PySM 2 models
        preset_strings : list of str
            List of strings identifiers for the models included in PySM 3,
            these are exactly the same models included in PySM 2, e.g.
            `["d2", "s1", "a1"]`, see the documentation for details about the
            available models.
        component_config : dict or TOML filename
            Modify the configuration of one of the included components or create
            a new component based on a Python dictionary or a TOML filename,
            see for example the TOML configuration file for the ``presets.cfg``
            file in the ``data`` folder of the package.
        component_config : list of Model subclasses
            List of component objects already initialized, typically subclasses of PySM.Model
            This is the most flexible way to provide a custom model to PySM
        output_unit : astropy Unit or string
            Astropy unit, e.g. "K_CMB", "MJ/sr"
        smoothing_angle : astropy.units.Quantity, optional
            Target output FWHM. When set, template components that support
            presmoothing (e.g. :class:`~pysm3.PowerLaw`) are built so that each
            amplitude template is smoothed by only the *differential* between
            this target and the presmoothing it already carries, rather than by
            the full target. See the documentation for details.
        map_dist: pysm.MapDistribution
            Distribution object used for parallel computing with MPI
        """

        if nside is None and not component_objects:  # not None and not []
            raise Exception("Need to specify nside in Sky")
        elif nside is None:
            nside = component_objects[0].nside
        elif component_objects:
            for comp in component_objects:
                assert (
                    nside == comp.nside
                ), "Component objects should have same NSIDE of Sky"

        super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        self.components = component_objects if component_objects is not None else []
        # otherwise instantiate the sky object from list of predefined models,
        # identified by their strings. These are defined in `pysm.presets`.
        if component_config is None:
            component_config = {}
        elif not isinstance(component_config, dict):
            component_config = toml.load(component_config)
        if preset_strings is not None:
            assert isinstance(preset_strings, list), "preset_strings should be a list"
            for string in preset_strings:
                component_config[string] = PRESET_MODELS[string]
        if len(component_config) > 0:
            self.components += create_components_from_config(
                component_config,
                nside=nside,
                map_dist=map_dist,
                smoothing_angle=smoothing_angle,
            )
        self.smoothing_angle = smoothing_angle
        self.output_unit = u.Unit(output_unit)

    def add_component(self, component):
        self.components.append(component)

    @property
    def includes_smoothing(self):
        return all(
            getattr(comp, "includes_smoothing", False) for comp in self.components
        )

    def get_emission(self, freq, weights=None, **kwargs):
        """This function returns the emission at a frequency, set of
        frequencies, or over a bandpass.
        """
        output = self.components[0].get_emission(freq, weights=weights, **kwargs)
        for comp in self.components[1:]:
            output += comp.get_emission(freq, weights=weights, **kwargs)
        return output * bandpass_unit_conversion(freq, weights, self.output_unit)
