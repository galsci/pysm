import warnings
import os

from astropy.utils import data

DATAURL = "https://portal.nersc.gov/project/cmb/pysm-data/"

PREDEFINED_DATA_FOLDERS = ["/global/project/projectdirs/cmb/www/pysm-data/"]  # NERSC


class RemoteData:
    def __init__(self):
        """Access template from remote server

        PySM input templates are stored on the CMB project space at NERSC
        and are made available via web.
        The get method of this class tries to access data locally from one
        of the PREDEFINED_DATA_FOLDERS defined above, if it fails, it
        retrieves the files and caches them remotely using facilities
        provided by `astropy.utils.data`.

        TODO: port from so_pysm_models also handling of templates in different
        coordinate systems
        """
        self.data_url = DATAURL
        self.data_folders = []
        try:
            self.data_folders.append(os.environ["PYSM_LOCAL_DATA"])
        except KeyError:
            pass
        self.data_folders += PREDEFINED_DATA_FOLDERS

    def get(self, filename):
        for folder in self.data_folders:
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                warnings.warn(f"Access data from {full_path}")
                return full_path
        with data.conf.set_temp("dataurl", self.data_url), data.conf.set_temp(
            "remote_timeout", 30
        ):
            warnings.warn(f"Retrieve data for {filename} (if not cached already)")
            full_path = data.get_pkg_data_filename(filename, show_progress=True)
        return full_path
