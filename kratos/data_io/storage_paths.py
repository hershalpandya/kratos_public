import socket
import logging
import os
from pathlib import Path

logger = logging.getLogger(os.path.basename(__file__))


class PathFinder:
    """
    Provides paths to storage locations on different machines (coma / t2b / personal_machine).

    Usage:
    .. code-block:: python
       path_finder = PathFinder()

       # if you are on coma:
       tbb_path = path_finder.get_radio_TBB_files_root_dir()

       # otherwise:
       path_finder.set_radio_TBB_files_root_dir("my_path")
       tbb_path = path_finder.get_radio_TBB_files_root_dir()

       # it is important to set the paths because this will be later in code.
    """

    def __init__(self):
        """
        Initialize PathFinder().
        """
        self.__event_plot_files_root_dir = None
        self.__metadata_dir = None
        self.__radio_TBB_files_root_dir = None
        self.__event_json_files_root_dir = None
        self.__particle_root_files_root_dir = None
        self.__event_hdf5_files_root_dir = None
        self.__auto_path_setting_succeeded = False

        self.set_file_paths_auto()

        if not self.__auto_path_setting_succeeded:
            logger.error(
                "Auto Setting of File Paths Failed. Use set() methods to manually do so."
            )

        return

    def set_file_paths_auto(self):
        """
        Looks up current hostname() i.e. machine name.
        Has default paths hardcoded for frequent use.
        """

        machine_name = socket.getfqdn()
        coma = "coma" in machine_name or "science.ru.nl" in machine_name
        t2b = "iihe.ac.be" in machine_name
        logger.info("Found machine name: %s" % machine_name)

        if coma and t2b:
            msg = "Machine name cannot be both coma and t2b."
            logger.error(msg)
            return

        if coma:
            logger.info("Setting paths for coma")
            self.__auto_path_setting_succeeded = True
            self.__radio_TBB_files_root_dir = "/vol/astro3/lofar/vhecr/lora_triggered/data/"
            self.__event_json_files_root_dir = "/vol/astro7/lofar/kratos_files/json"
            self.__particle_root_files_root_dir = "/vol/astro5/lofar/vhecr/lora_triggered/LORAraw/"
            self.__event_hdf5_files_root_dir = "/vol/astro7/lofar/kratos_files/h5/"
            self.__event_plot_files_root_dir = "/vol/astro7/lofar/kratos_files/plots"
            self.__metadata_dir = "/vol/astro7/lofar/vhecr/kratos/data/"
        elif t2b:
            # TODO: add paths for T2B
            pass
        else:
            logger.warning("No preset paths for this machine name!")
            # TODO: add default local dirs
            self.__metadata_dir = "../../data"
        # source_path = Path(__file__).resolve()
        # source_dir = source_path.parent

        # self.__metadata_dir = '../../data' # CHANGE to abs path to source (not installed modules in .../site-packages/...)
        logger.debug("Metadata dir set to: %s" % self.__metadata_dir)
        return

    def get_particle_root_files_root_dir(self):
        return self.__particle_root_files_root_dir
    
    def get_radio_TBB_files_root_dir(self):
        return self.__radio_TBB_files_root_dir

    def get_event_json_files_root_dir(self):
        return self.__event_json_files_root_dir

    def get_event_hdf5_files_root_dir(self):
        return self.__event_hdf5_files_root_dir

    def get_event_plot_files_root_dir(self):
        return self.__event_plot_files_root_dir

    def get_metadata_dir(self):
        return self.__metadata_dir

    def set_radio_TBB_files_root_dir(self, path):
        self.__radio_TBB_files_root_dir = path
        return

    def set_event_json_files_root_dir(self, path):
        self.__event_json_files_root_dir = path
        return

    def set_event_hdf5_files_root_dir(self, path):
        self.__event_json_files_root_dir = path
        return

    def set_event_plot_files_root_dir(self, path):
        self.__event_plot_files_root_dir = path
        return

    def set_metadata_dir(self, path):
        self.__metadata_dir = path
        return
