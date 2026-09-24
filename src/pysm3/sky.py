""" This submoduls contains most of the user-facing code, as the objects
`Sky` and `Instrument` can be used for most of the functionality of the
code, without delving into the difference `Model` objects.

Objects:
    Sky
"""

import toml
from . import units as u
from .models import *
from .models import Model, parse_pre_applied_fwhm
from .utils import bandpass_unit_conversion


def remove_class_from_dict(d):
    """Return a copy of dictionary without the key "class" """
    return {k: d[k] for k in d if k != "class"}


def create_component_from_config(config, nside, map_dist=None):
    """Create one component from its configuration dictionary

    The ``pre_applied_fwhm`` keyword is handled generically for all
    components, independently of their constructor signature: it is
    removed from the configuration, parsed and set as attribute of the
    created component, where the `Model` base class uses it to tag the
    output of `get_emission`.
    """
    component_class = globals()[config["class"]]
    kwargs = remove_class_from_dict(config)
    pre_applied_fwhm = kwargs.pop("pre_applied_fwhm", None)
    component = component_class(**kwargs, nside=nside, map_dist=map_dist)
    if pre_applied_fwhm is not None:
        component.pre_applied_fwhm = parse_pre_applied_fwhm(pre_applied_fwhm)
    return component


def get_common_pre_applied_fwhm(components):
    """Return the common pre_applied_fwhm of a list of components

    Raises a ValueError if some components declare a pre-applied beam and
    others do not, or if they declare different values: the summed emission
    would not have a single well-defined beam and any smoothing would be
    wrong for at least one component.
    """
    values = [getattr(comp, "pre_applied_fwhm", None) for comp in components]
    if any(value is not None for value in values):
        canonical = [
            None if value is None else u.Quantity(value).to(u.rad).value
            for value in values
        ]
        if None in canonical or len(set(canonical)) > 1:
            raise ValueError(
                "All components of a Sky must have the same pre_applied_fwhm "
                f"(or none at all), got: {values}. Combine components "
                "delivered at the same resolution, or create and smooth "
                "separate Sky objects"
            )
    return values[0] if values else None


def create_components_from_config(config, nside, map_dist=None):
    """Create a list of components from their configuration

    A configuration either describes a single component (with a "class"
    key), or is a dictionary of named component configurations, which in
    turn may each be a single component or a nested collection.
    """
    output_components = []
    if "class" in config:
        output_components.append(
            create_component_from_config(config, nside=nside, map_dist=map_dist)
        )
        return output_components

    for model_config in config.values():
        if "class" in model_config:
            output_component = create_component_from_config(
                model_config, nside=nside, map_dist=map_dist
            )
        else:  # multiple components
            partial_components = [
                create_component_from_config(
                    each_config, nside=nside, map_dist=map_dist
                )
                for each_config in model_config.values()
            ]
            output_component = Sky(
                component_objects=partial_components, nside=nside, map_dist=map_dist
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
    All the components of a multi-component Sky must have the same
    ``pre_applied_fwhm`` (or none at all), see the documentation about
    the pre-applied beam, otherwise a ``ValueError`` is raised at
    creation.
    See the tutorials section of the documentation for examples.

    Attributes
    ----------
    components: list(pysm.Model object)
        List of `pysm.Model` objects.
    pre_applied_fwhm: astropy.units.Quantity or None
        Beam already applied to the templates of all the components,
        attached to the maps returned by `get_emission`.
    """

    def __init__(
        self,
        nside=None,
        max_nside=None,
        preset_strings=None,
        component_config=None,
        component_objects=None,
        output_unit=u.uK_RJ,
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
                component_config, nside=nside, map_dist=map_dist
            )
        self.output_unit = u.Unit(output_unit)
        self.pre_applied_fwhm = get_common_pre_applied_fwhm(self.components)

    def add_component(self, component):
        self.pre_applied_fwhm = get_common_pre_applied_fwhm(
            self.components + [component]
        )
        self.components.append(component)

    @property
    def includes_smoothing(self):
        return all(
            getattr(comp, "includes_smoothing", False) for comp in self.components
        )

    def get_emission(self, freq, weights=None, **kwargs):
        """This function returns the emission at a frequency, set of
        frequencies, or over a bandpass.

        The output map carries the ``pre_applied_fwhm`` common to all
        components, if any, attached by the `Model` base class.
        """
        output = self.components[0].get_emission(freq, weights=weights, **kwargs)
        for comp in self.components[1:]:
            output += comp.get_emission(freq, weights=weights, **kwargs)
        return output * bandpass_unit_conversion(freq, weights, self.output_unit)
