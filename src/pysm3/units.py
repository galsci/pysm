""" Submodule containing definitions of units used in PySM
that are not already contained within `astropy.units`. These are

- K_RJ: Rayleigh-Jeans temperature
- K_CMB: thermodynamic temperature

We use the factory functions of `astropy.units` to define units
that will work with the rest of astropy.
"""
from astropy.units import *  # noqa: F403

# Define new thermodynamic and Rayleigh-Jeans units. This
# is a little unsatisfying, as astropy.units.Quantity will
# only recognize the prefixes add by hand in `add_enabled_units`.
# Would be good to find away of automatically adding all SI prefixes
# to be recognized by astropy.
def_unit(
    r"K_CMB",
    namespace=globals(),
    prefixes=True,
    doc="Kelvin CMB: Thermodynamic temperature units.",
    format={"generic": r"K_CMB", "latex": "K_{{CMB}}"},
)

def_unit(
    r"K_RJ",
    namespace=globals(),
    prefixes=True,
    doc="Kelvin Rayleigh-Jeans:  brightness temperature.",
    format={"generic": r"K_RJ", "latex": "K_{{RJ}}"},
)


@quantity_input(equivalencies=spectral())
def cmb_equivalencies(spec: GHz):
    """ Function defining the conversion between RJ, thermodynamic,
    and flux units.

    Parameters
    ----------
    spec: `astropy.Quantity`
        Spectral quantity that may be converted to frequency. Frequency at which
        the conversion is to be calculated.

    Returns
    -------
    list(tuple(unit_from, unit_to, forward, backward))
        Returns a list of unit equivalencies, which are tuples containing the
        units from and to which the conversion is applied, and the forward
        and backward transformations.
    """
    nu = spec.to(GHz, equivalencies=spectral())
    # use the equivalencies for thermodynamic and RJ units already
    # contained within astropy for conversion between spectral radiance
    # and temperature.
    try:
        [(_, _, Jy_to_CMB, CMB_to_Jy)] = thermodynamic_temperature(nu)
    except NameError:
        raise
    [(_, _, Jy_to_RJ, RJ_to_Jy)] = brightness_temperature(nu)

    def RJ_to_CMB(T_RJ):
        return Jy_to_CMB(RJ_to_Jy(T_RJ))

    def CMB_to_RJ(T_CMB):
        return Jy_to_RJ(CMB_to_Jy(T_CMB))

    return [
        (K_RJ, K_CMB, RJ_to_CMB, CMB_to_RJ),
        (Jy / sr, K_RJ, Jy_to_RJ, RJ_to_Jy),
        (Jy / sr, K_CMB, Jy_to_CMB, CMB_to_Jy),
    ]


add_enabled_units([uK_RJ, mK_RJ, K_RJ, kK_RJ, MK_RJ])
add_enabled_units([uK_CMB, mK_CMB, K_CMB, kK_CMB, MK_CMB])
