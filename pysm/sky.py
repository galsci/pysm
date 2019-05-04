""" This submoduls contains most of the user-facing code, as the objects
`Sky` and `Instrument` can be used for most of the functionality of the
code, without delving into the difference `Model` objects.

Objects:
    Sky
"""
import numpy as np
from .models import Model


class Sky(Model):
    """ This class is a convenience object that wraps together a group of
    component models. It acts like a single `pysm.Model` object, in that it
    is sub-classed from the `pysm.Model` template, and therefore has the same
    functionality.
    
    Attributes
    ----------
    components: list(pysm.Model object)
        List of `pysm.Model` objects.
    """

    def __init__(
        self,
        nside=None,
        pixel_indices=None,
        component_objects=None,
        preset_strings=None,
        mpi_comm=None,
    ):
        super().__init__(nside=nside, pixel_indices=pixel_indices, mpi_comm=mpi_comm)
        if component_objects is not None:
            self.components = component_objects
        # otherwise instantiate the sky object from list of predefined models,
        # identified by their strings. These are defined in `pysm.presets`.
        if preset_strings is not None:
            # as `pysm.presets` contains an import of `pysm.Sky`, importing here
            # limits the circular nature of these imports.
            from .presets import preset_models

            try:
                assert isinstance(preset_strings, list)
            except AssertionError:
                print(
                    """pysm.Sky may take list of model strings when instantiated,
                check input."""
                )
                raise
            self.components = [
                preset_models(string, nside) for string in preset_strings
            ]
        return

    def get_emission(self, nu):
        """ This function returns the emission at a frequency, set of
        frequencies, or over a bandpass.
        """
        return sum([comp.get_emission(nu) for comp in self.components])
