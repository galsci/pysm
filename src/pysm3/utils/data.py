import contextlib
import logging
import os
from urllib.error import URLError

from astropy.utils import data

DATAURL = "https://portal.nersc.gov/project/cmb/pysm-data/"

PREDEFINED_DATA_FOLDERS = ["/global/cfs/cdirs/cmb/www/pysm-data/"]  # NERSC

log = logging.getLogger("pysm3")


class RemoteData:
    def __init__(self):
        """Access template from remote server

        PySM input templates are stored on the CMB project space at NERSC
        and are made available via web.
        The get method of this class tries to access data locally from one
        of the PREDEFINED_DATA_FOLDERS defined above or directly if given an absolute path.
        If it fails, it
        retrieves the files and caches them remotely using facilities
        provided by `astropy.utils.data`.

        TODO: port from so_pysm_models also handling of templates in different
        coordinate systems
        """
        self.data_url = DATAURL
        self.data_folders = []
        with contextlib.suppress(KeyError):
            self.data_folders.append(os.environ["PYSM_LOCAL_DATA"])
        self.data_folders += PREDEFINED_DATA_FOLDERS

    def get(self, filename):
        if os.path.exists(filename):
            log.info(f"Access data from {filename}")
            return filename
        for folder in self.data_folders:
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                log.info(f"Access data from {full_path}")
                return full_path
        with data.conf.set_temp("dataurl", self.data_url), data.conf.set_temp(
            "remote_timeout", 90
        ):
            log.info(f"Retrieve data for {filename} (if not cached already)")
            try:
                full_path = data.get_pkg_data_filename(filename, show_progress=True)
            except URLError as e:
                log.error(
                    "File not found, please make sure you are using the latest version of PySM 3"
                )
                raise e
        return full_path
