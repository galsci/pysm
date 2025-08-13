import pytest
import numpy as np
import healpy as hp
from pysm3 import units as u
from pysm3 import Sky
from pysm3.models.template import read_map

# Reference data
NSIDE = 256
FREQUENCIES = [0, 80, 90, 100, 110] * u.GHz
REFERENCE_URLS = {
    0 * u.GHz: "test_data/dipole/dipole_nside0256_freq000GHz.fits",
    80 * u.GHz: "test_data/dipole/dipole_nside0256_freq080GHz.fits",
    90 * u.GHz: "test_data/dipole/dipole_nside0256_freq090GHz.fits",
    100 * u.GHz: "test_data/dipole/dipole_nside0256_freq100GHz.fits",
    110 * u.GHz: "test_data/dipole/dipole_nside0256_freq110GHz.fits",
}


def get_quadrupole_and_dipole_amplitudes(hmap, lmax=2):
    """
    Calculates the quadrupole and dipole amplitudes from a HEALPix map.
    Assumes the map is a temperature map (I).
    """
    alm = hp.map2alm(hmap, lmax=lmax)
    cl = hp.alm2cl(alm, lmax=lmax)
    C2 = cl[2]
    quadrupole_amplitude = np.sqrt(C2)

    # Calculate dipole amplitude
    mono, vec = hp.fit_dipole(hmap, gal_cut=10)  # gal_cut to avoid galactic plane
    dipole_amplitude = np.sqrt(np.sum(vec ** 2))

    return quadrupole_amplitude, dipole_amplitude


@pytest.mark.parametrize("freq", [80, 90, 100, 110] * u.GHz)
def test_quadrupole_corrected_freqs(freq):
    # Load reference map
    reference_map = read_map(REFERENCE_URLS[freq], nside=NSIDE)

    # Instantiate Sky with dip2 (quadrupole correction enabled)
    dipole_model = Sky(nside=NSIDE, preset_strings=["dip2"])

    # Generate dipole map
    generated_map = dipole_model.get_emission(freq)

    # Convert generated map to K_CMB for comparison
    generated_map_K_CMB = generated_map.to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freq)
    )

    # Compare maps
    np.testing.assert_allclose(
        generated_map_K_CMB.value, reference_map.value, rtol=1e-6, atol=1e-6
    )


def test_no_quadrupole():
    # Load 0 GHz reference map
    ref_0ghz_map = read_map(REFERENCE_URLS[0 * u.GHz], nside=NSIDE)

    # Instantiate Sky with dip1 (no quadrupole correction)
    dipole_model = Sky(nside=NSIDE, preset_strings=["dip1"])

    # Generate map at 100 GHz with no quadrupole correction
    generated_100ghz_map = dipole_model.get_emission(100 * u.GHz)

    # Convert generated map to K_CMB for comparison
    generated_100ghz_map_K_CMB = generated_100ghz_map.to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(100 * u.GHz)
    )

    # Compare maps
    np.testing.assert_allclose(
        generated_100ghz_map_K_CMB.value, ref_0ghz_map.value, rtol=1e-6, atol=1e-6
    )


def calculate_relative_difference(val1, val2):
    if np.isclose(val2, 0.0):
        return "N/A"  # Avoid division by zero
    return (val1 - val2) / val2


def test_print_quadrupole_amplitudes():
    frequencies_to_test = [80, 90, 100, 110] * u.GHz

    # Data storage
    quadrupole_data = []
    dipole_data = []

    # Load 0 GHz reference map for TOAST comparison
    ref_0ghz_map = read_map(REFERENCE_URLS[0 * u.GHz], nside=NSIDE)
    toast_0ghz_quad_amp, toast_0ghz_dipole_amp = get_quadrupole_and_dipole_amplitudes(
        ref_0ghz_map.value
    )

    # PySM No Quadrupole (no quadrupole correction) - constant across frequencies
    dipole_no_quad_model = Sky(nside=NSIDE, preset_strings=["dip1"])
    map_no_quad_uK_RJ = dipole_no_quad_model.get_emission(
        frequencies_to_test[0]
    )  # Use first freq for emission
    map_no_quad_K_CMB = map_no_quad_uK_RJ.to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(frequencies_to_test[0])
    )
    (
        pysm_no_quad_amp_constant,
        pysm_no_quad_dipole_amp_constant,
    ) = get_quadrupole_and_dipole_amplitudes(map_no_quad_K_CMB.value)

    for freq in frequencies_to_test:
        # Load reference map for current frequency (TOAST map)
        reference_map = read_map(REFERENCE_URLS[freq], nside=NSIDE)
        (
            toast_current_freq_quad_amp,
            toast_current_freq_dipole_amp,
        ) = get_quadrupole_and_dipole_amplitudes(reference_map.value)

        # PySM Quadrupole (quadrupole correction enabled)
        dipole_quad_model = Sky(nside=NSIDE, preset_strings=["dip2"])
        map_quad_uK_RJ = dipole_quad_model.get_emission(freq)
        map_quad_K_CMB = map_quad_uK_RJ.to(
            u.K_CMB, equivalencies=u.cmb_equivalencies(freq)
        )
        pysm_quad_amp, pysm_quad_dipole_amp = get_quadrupole_and_dipole_amplitudes(
            map_quad_K_CMB.value
        )

        # Store data for tables
        quadrupole_data.append(
            {
                "freq": freq,
                "pysm_quad": pysm_quad_amp,
                "toast_current_freq_quad": toast_current_freq_quad_amp,
            }
        )
        dipole_data.append(
            {
                "freq": freq,
                "pysm_quad_dipole": pysm_quad_dipole_amp,
                "toast_current_freq_dipole": toast_current_freq_dipole_amp,
            }
        )

    # Print Quadrupole Table
    print()
    print("# Quadrupole Amplitudes (K_CMB)")
    print("| Frequency | PySM Quad | TOAST | Rel Diff (PySM Quad vs TOAST) |")
    print("|---|---|---|---|---|")
    for row in quadrupole_data:
        rel_diff_pysm_quad_vs_toast = calculate_relative_difference(
            row["pysm_quad"], row["toast_current_freq_quad"]
        )
        print(
            f"| {row['freq']} | {row['pysm_quad']:.4e} | {row['toast_current_freq_quad']:.4e} | {rel_diff_pysm_quad_vs_toast:.4e} |"
        )
    # No Quadrupole row
    rel_diff_pysm_no_quad_vs_toast_0ghz = calculate_relative_difference(
        pysm_no_quad_amp_constant, toast_0ghz_quad_amp
    )
    print(
        f"| No Quad | {pysm_no_quad_amp_constant:.4e} | {toast_0ghz_quad_amp:.4e} | {toast_0ghz_quad_amp:.4e} | {rel_diff_pysm_no_quad_vs_toast_0ghz:.4e} |"
    )

    # Print Dipole Table
    print()
    print()
    print("# Dipole Amplitudes (K_CMB)")
    print(
        "| Frequency | PySM Quad Dipole | TOAST Dipole | Rel Diff (PySM Quad Dipole vs TOAST) |"
    )
    print("|---|---|---|---|---|")
    for row in dipole_data:
        rel_diff_pysm_quad_dipole_vs_toast = calculate_relative_difference(
            row["pysm_quad_dipole"], row["toast_current_freq_dipole"]
        )
        print(
            f"| {row['freq']} | {row['pysm_quad_dipole']:.4e} | {row['toast_current_freq_dipole']:.4e} | {rel_diff_pysm_quad_dipole_vs_toast:.4e} |"
        )
    # No Quadrupole Dipole row
    rel_diff_pysm_no_quad_dipole_vs_toast_0ghz = calculate_relative_difference(
        pysm_no_quad_dipole_amp_constant, toast_0ghz_dipole_amp
    )
    print(
        f"| No Quad | {pysm_no_quad_dipole_amp_constant:.4e} | {toast_0ghz_dipole_amp:.4e} | {rel_diff_pysm_no_quad_dipole_vs_toast_0ghz:.4e} |"
    )
