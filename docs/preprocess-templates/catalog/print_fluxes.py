import argparse
import xarray as xr
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Print fluxes of the first and last 10 sources from a WebSky catalog."
    )
    parser.add_argument(
        "catalog_path",
        type=str,
        help="Path to the output catalog HDF5 file (e.g., websky_high_flux_catalog_1mJy.h5)",
    )
    args = parser.parse_args()

    try:
        output_catalog = xr.open_dataset(args.catalog_path)
    except FileNotFoundError:
        print(f"Error: Catalog file not found at {args.catalog_path}")
        return
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return

    # Calculate flux at 100 GHz using the polynomial coefficients
    # The coefficients are stored in (power, index) order, with highest degree first
    log_freq_100 = np.log(100)
    flux_100 = np.polyval(output_catalog["logpolycoefflux"].values, log_freq_100)

    # Add flux_100 as a new variable to the dataset for sorting
    output_catalog["flux_100"] = ("index", flux_100)

    print("Fluxes of the first 10 sources (at 100 GHz):")
    for i in range(10):
        print(f"Source {i+1}: {output_catalog['flux_100'].isel(index=i).item():.4e} Jy")

    print("\nFluxes of the last 10 sources (at 100 GHz):")
    for i in range(1, 11):
        print(
            f"Source {len(output_catalog['flux_100']) - i + 1}: {output_catalog['flux_100'].isel(index=-i).item():.4e} Jy"
        )


if __name__ == "__main__":
    main()
