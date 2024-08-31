import numpy as np
from numba import njit


@njit(parallel=False)
def map_to_log_pol_tens(m):
    """Transform from IQU to Log pol tensor formalism

    See the notebook about preprocessing GNILC maps in
    the documentation of PySM 3 for more details"""
    P = np.sqrt(m[1] ** 2 + m[2] ** 2)
    log_pol_tens = np.empty_like(m)
    log_pol_tens[0] = np.log(m[0] ** 2 - P**2) / 2.0
    log_pol_tens[1:] = m[1:] / P * np.log((m[0] + P) / (m[0] - P)) / 2.0
    return log_pol_tens


@njit(parallel=False)
def log_pol_tens_to_map(log_pol_tens):
    """Transform from Log pol tensor formalism to IQU

    See the notebook about preprocessing GNILC maps in
    the documentation of PySM 3 for more details"""
    P = np.sqrt(log_pol_tens[1] ** 2 + log_pol_tens[2] ** 2)
    m = np.empty_like(log_pol_tens)
    exp_i = np.exp(log_pol_tens[0])
    m[0] = exp_i * np.cosh(P)
    m[1:] = log_pol_tens[1:] / P * exp_i * np.sinh(P)
    return m
