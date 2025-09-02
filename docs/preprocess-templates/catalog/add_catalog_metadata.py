import xarray as xr
import os

def add_galactic_metadata_to_catalog(catalog_filepath):
    """
    Opens an existing catalog file, adds 'ref_frame' metadata as 'Galactic',
    and saves the modified catalog back to the same file.

    Parameters
    ----------
    catalog_filepath : str
        The absolute path to the catalog file (e.g., 'websky_high_flux_catalog_1mJy.h5').
    """
    try:
        # Open the existing catalog dataset
        print(f"Opening catalog: {catalog_filepath}")
        catalog = xr.open_dataset(catalog_filepath)

        # Add the 'ref_frame' attribute to the dataset's metadata
        catalog.attrs["ref_frame"] = "Galactic"
        print("Added 'ref_frame: Galactic' metadata.")

        # Save the modified catalog back to the same file in NETCDF4 format (which is HDF5 based)
        print(f"Saving modified catalog to: {catalog_filepath}")
        catalog.to_netcdf(catalog_filepath, format="NETCDF4")
        print("Catalog saved successfully.")

    except FileNotFoundError:
        print(f"Error: The catalog file was not found at '{catalog_filepath}'.")
        print("Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual absolute path to your catalog file.
    # Based on the notebook, the file is likely located here:
    catalog_file_path = "/global/cfs/cdirs/sobs/www/users/Radio_WebSky/matched_catalogs_2/websky_high_flux_catalog_1mJy.h5"

    add_galactic_metadata_to_catalog(catalog_file_path)
