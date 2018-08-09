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

def data_path(fname):
    """ Function to add a given file name to the directory in which the
    PySM data is stored.

    Parameters
    ----------
    fname: string
        String, file name of a data product used by PySM.

    Returns
    -------
    `pathlib.Path` object
        Instance of the `Path` object.
    """
    data_dir = Path(__file__).absolute().parent / 'data'
    return data_dir / fname

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
    if model_string == 'd1':
        config = {
            'map_I': data_path('dust_t_new.fits'),
            'map_Q': data_path('dust_q_new.fits'),
            'map_U': data_path('dust_u_new.fits'),
            'map_mbb_index': data_path('dust_beta.fits'),
            'map_mbb_temperature': data_path('dust_temp.fits'),
            'freq_ref_I': 545.,
            'freq_ref_P': 353.,
            'nside': nside,
        }
        return ModifiedBlackBody(**config)

    if model_string == 'd4':
        config_comp1 = {
            'map_I': data_path('dust2comp_I1_ns512_545.fits'),
            'map_Q': data_path('dust2comp_Q1_ns512_353.fits'),
            'map_U': data_path('dust2comp_U1_ns512_353.fits'),
            'map_mbb_index': data_path('dust2comp_beta1_ns512.fits'),
            'map_mbb_temperature': data_path('dust2comp_temp1_ns512.fits'),
            'freq_ref_I': 545.,
            'freq_ref_P': 353.,
            'nside': nside,
        }
        config_comp2 = {
            'map_I': data_path('dust2comp_I2_ns512_545.fits'),
            'map_Q': data_path('dust2comp_Q2_ns512_353.fits'),
            'map_U': data_path('dust2comp_U2_ns512_353.fits'),
            'map_mbb_index': data_path('dust2comp_beta2_ns512.fits'),
            'map_mbb_temperature': data_path('dust2comp_temp2_ns512.fits'),
            'freq_ref_I': 545.,
            'freq_ref_P': 353.,
            'nside': nside,
        }
        # note that in this model we are using the `pysm.Sky` object as a
        # container to combine two separate instances of the
        # `pysm.ModifiedBlackBody` object.
        mbb1 = ModifiedBlackBody(**config_comp1)
        mbb2 = ModifiedBlackBody(**config_comp2)
        return Sky(component_objects=[mbb1, mbb2])

    if model_string == 'd6':
        config = {
            'correlation_length': 5.0,
            'map_I': data_path('dust_t_new.fits'),
            'map_Q': data_path('dust_q_new.fits'),
            'map_U': data_path('dust_u_new.fits'),
            'map_mbb_index': data_path('dust_beta.fits'),
            'map_mbb_temperature': data_path('dust_temp.fits'),
            'freq_ref_I': 545.,
            'freq_ref_P': 353.,
            'nside': nside,
        }
        return DecorrelatedModifiedBlackBody(**config)

    if model_string == 's1':
        config = {
            'map_I': data_path('synch_t_new.fits'),
            'map_Q': data_path('synch_q_new.fits'),
            'map_U': data_path('synch_u_new.fits'),
            'map_pl_index': data_path('synch_beta.fits'),
            'freq_ref_I': 0.408,
            'freq_ref_P': 23.,
            'nside': nside,
        }
        return SynchrotronPowerLaw(**config)
    return None
