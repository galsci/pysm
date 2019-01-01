""" This submodule contains the code used for the units
system in PySM. We use the `astropy.units` module for
the infrastructure of unit conversion. We add a unit,
thermodynamic temperature, which is not in the standard
set of units included with the package.
"""
from astropy.units import *

# Define new thermodynamic and Rayleigh-Jeans units. This
# is a little unsatisfying, as astropy.units.Quantity will
# only recognize the prefixes add by hand in `add_enabled_units`.
# Would be good to find away of automatically adding all SI prefixes
# to be recognized by astropy.
def_unit(r'K_CMB',
         namespace=globals(),
         prefixes=True,
         doc='Kelvin CMB: Thermodynamic temperature units.',
         format={
             'generic': r'K_CMB',
             'latex': 'K_{{CMB}}'},
         
) 

def_unit(r'K_RJ',
         namespace=globals(),
         prefixes=True,
         doc='Kelvin Rayleigh-Jeans:  brightness temperature.',
         format={
             'generic': r'K_RJ',
             'latex': 'K_{{RJ}}'
         },
)

@quantity_input(equivalencies=spectral())
def cmb_equivalencies(spec):
    nu = spec.to(GHz, equivalencies=spectral())
    [(_, _, Jy_to_CMB, CMB_to_Jy)] = thermodynamic_temperature(nu)
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
