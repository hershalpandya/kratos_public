import glob
import logging
import os

import numpy as np

from kratos.station import Station
from kratos.data_io import DBManager
from kratos.data_io import HDFWriterReader
from kratos.data_io import lofar_io

logger = logging.getLogger(os.path.basename(__file__))


class Event:
    # (AC) TODO: drop option for giving a different db_storage_path and tbb_storage_path ??
    # Seems only confusing, should have them in storage_paths, not here.

    # (HP) : somewhere, we should have a possibility of giving custom paths.
    # E.g. if you run it on your laptop, where do you set PathFinder paths?
    def __init__(self,
                 pathfinder=None,
                 event_id=None,
                 ):
        """
        If event_id provided, loads event info from json file.
        Else creates empty variables.

        :param pathfinder: instance of class PathFinder() with paths set.
        :type pathfinder: PathFinder()
        :param event_id: unique event identifier. default:None.
        :type event_id: int
        """

        if not pathfinder:
            logger.error("PathFinder() not provided.")
            raise Exception

        self.event_dict = None
        self.__db_storage_path = pathfinder.get_event_json_files_root_dir()
        self.__tbb_storage_path = pathfinder.get_radio_TBB_files_root_dir()
        self.__h5_storage_path = pathfinder.get_event_hdf5_files_root_dir()
        self.__metadata_dir = pathfinder.get_metadata_dir()
        self.__json_storage_dir = pathfinder.get_event_json_files_root_dir()

        self.load_event_from_database(event_id)

        # list of station objects
        self.stations: list[Station] = []

    def load_LOFAR_stations(self):
        """From the tbb storage path, it will load all"""
        # lora time stamp has to be available.
        # use helper to find TBB files
        # give each TBB file to station.load_LOFAR()
        # make a list of station objects.
        tbb_fname = lofar_io.tbb_filetag_from_utc(
            self.event_dict["LORA"]["utc_time_stamp"]
        )

        tbb_fname = self.__tbb_storage_path + "/*" + tbb_fname + "*.h5"
        tbb_fs = glob.glob(
            tbb_fname
        )  # this is expensive in a big NFS-mounted directory...

        # add filenames to dictionary.
        assert len(tbb_fs) > 0, "no files found here: " + tbb_fname

        logger.info("Found %i TBB files for this event: " % len(tbb_fs))

        for tbb_filename in tbb_fs:
            logger.info(tbb_filename)
            # TODO: get filenames for the same station together
            # (when multiple files for one station)
            # then change the docstring for data_filename in station init.
            # have directory in Event, pass it down
            this_station = Station(data_filename=[tbb_filename], metadata_dir=self.__metadata_dir)

            logger.info("Reading in file (antennas) metadata")
            this_station.load_LOFAR_antennas()
            self.stations.append(this_station)
        return

    def load_LOFAR_traces(self, trace_length_nbins=65536):
        time_s = self.event_dict["LORA"]["utc_time_stamp"]
        time_ns = self.event_dict["LORA"]["time_stamp_ns"]

        stations_to_remove = []
        for this_station in self.stations:
            logger.debug(f"Reading in traces for station {this_station.get_station_name()}")
            try:
                this_station.load_LOFAR_traces(time_s, time_ns, trace_length_nbins)
            except ValueError:
                stations_to_remove.append(this_station)
                logger.warning(
                    f"Event not in file for station {this_station.get_station_name()}"
                )

        for station in stations_to_remove:
            self.stations.remove(station)

    def get_station_by_name(self, station_name):
        """
        Get the Station object from the Event station list by name.

        :param station_name: The name of the Station
        :type station_name: str
        """
        for station_obj in self.stations:
            if station_obj.get_station_name() == station_name:
                return station_obj

        logger.error(f"Cannot find any station with name:{station_name}")

        return None

    def get_data_dict_from_members(self):
        """
        Returns a dictionary with keys and values that have to be saved to disk.

        :return: dictionary with event keys and values
        :rtype: dict
        """
        data_dict = {'event_id': self.event_dict['event_id'],
                     'date_time': self.event_dict['date_time'],
                     'best_core_x_m': self.event_dict['best_core_x_m'],
                     'best_core_y_m': self.event_dict['best_core_y_m'],
                     'best_zenith_rad': self.event_dict['best_zenith_rad'],
                     'best_azimuth_rad': self.event_dict['best_azimuth_rad'],
                     'best_energy_GeV': self.event_dict['best_energy_GeV'],
                     'LORA': self.event_dict['LORA'],
                     'LOFAR': {}
                     }

        for i, sta in enumerate(self.stations):
            data_dict['LOFAR'][f"Station_{i}"] = sta.get_data_dict_from_members()

        return data_dict

    def set_members_from_data_dict(self, data_dict):
        """
        Accepts a dictionary as input and sets values for members of the class.
        Does not load event_dict.

        :param data_dict: nested dictionary with event keys and station keys
        :type data_dict: dict
        """

        if data_dict['event_id'] != self.event_dict['event_id']:
            logger.error("Event Id values from data_dict(hdf5) \
             and self.event_dict(json) are not same.")
            raise ValueError

        self.stations = []
        station_keys = [i for i in list(data_dict['LOFAR'].keys()) if 'Station_' in i]

        for key in station_keys:
            new_sta = Station(data_dict=data_dict['LOFAR'][key], metadata_dir=self.__metadata_dir)
            self.stations.append(new_sta)
        return

    def save_event_to_h5(self, fname=None):
        """
        Save event data to hdf5 file.

        :param fname: Full path filename.
        By default saves <event_id>.h5 at dir given by PathFinder.
        :type fname: str
        """
        if fname is None:
            fname = os.path.join(self.__h5_storage_path,
                                 str(self.event_dict['event_id']) + '.h5'
                                 )

        logger.info(f"Saving event to... {fname}")
        h5rw = HDFWriterReader()
        data_dict = self.get_data_dict_from_members()
        h5rw.write_event_dictionary_to_hdf(data_dict, fname)
        return

    def save_event_to_database(self):
        """self.event_dict is written to disk as json file."""
        db = DBManager()
        db.write_event_dictionary_to_disk(self.event_dict)
        return

    def load_event_from_database(self, event_id):
        """self.event_dict is loaded from json file."""

        db = DBManager(path=self.__json_storage_dir)
        if event_id:
            self.event_dict = db.get_event_dictionary_from_json_file(event_id)
        else:
            self.event_dict = db.get_event_dictionary_empty()
        return

    def load_event_from_h5(self, fname=None):
        """
        Load event data from hdf5 file.

        :param fname: Full path filename.
        By default loads <event_id>.h5 at dir given by PathFinder.
        :type fname: str
        """
        if fname is None:
            fname = os.path.join(self.__h5_storage_path,
                                 str(self.event_dict['event_id']) + '.h5'
                                 )
        h5rw = HDFWriterReader()
        data_dict = h5rw.read_hdf_to_dictionary(fname)
        print(data_dict.keys())
        self.set_members_from_data_dict(data_dict)
        return

    def trim_traces(self, new_tracelength_nbins):
        """
        Trims all traces. Time_array, Raw, Clean, Calibrated, Ex,Ey, and Ez.

        Trims traces for all stations to new value.
        Space saving technique. To be called after RFI cleaning.

        :param new_tracelength_nbins: trace length to store in h5 files.
        :type new_tracelength_nbins: int
        """
        for sta in self.stations:
            sta.trim_traces(new_tracelength_nbins)
        return

    def apply_filter(self, filter_object, trace_type='raw', save_type=None):
        """
        Apply a given Filter to all traces in all stations present in the Event.
        """
        if save_type is None:
            save_type = trace_type

        for st in self.stations:
            st.apply_to_all_traces(filter_object.apply_to_trace,
                                   trace_type=trace_type, save_type=save_type)

    def do_rfi_cleaning(self):
        """
        Asks each Station() instance to do its rfi_cleaning.
        """
        for sta in self.stations:
            sta.do_rfi_cleaning()
        return

    def do_galactic_calibration(self, experiment_name, cal_type='new'):
        """
        Asks each Station() instance to do its galactic calib.
        """
        if not self.event_dict['event_id']:
            logger.error(f"event_id is None. Galactic Calib Fails.")
            raise ValueError

        # Calculate UNIX timestamp based on event ID
        timestamp = int(self.event_dict['event_id']) + 1262304000
        for sta in self.stations:
            sta.do_galactic_calibration(timestamp, experiment_name, cal=cal_type)
        return

    def do_antenna_unfolding(self):
        """
        Asks each Station() instance to unfold all its antennas.
        """
        if not self.event_dict['event_id']:
            logger.error(f"event_id is None. Antenna Unfolding Fails.")
            raise ValueError

        for sta in self.stations:
            sta.do_antenna_unfolding(self)
        return

    def do_apply_calibration_delays(self, trace_type='calibrated'):
        """
        Apply the extracted calibration delays to each station in the event.

        :param trace_type: The trace to which to apply the delays, defaults to 'calibrated'.
        :type trace_type: str
        """
        # TODO: check if this syntax works for all stations
        for st in self.stations:
            st.apply_calibration_delays(trace_type=trace_type, save_type=trace_type)

    def get_cleaned_dipole_trace(self, station_name, antenna_id, polarization):
        """
        :param station_name: The name of the station.
        :type station_name: str
        :param antenna_id: antenna id ("a"+str(even_dipole_id))
        :type antenna_id: str
        :param polarization: pol value 0/1
        :type polarization: int
        """
        sta = self.get_station_by_name(station_name)
        ant = sta.get_antenna_by_id(antenna_id)
        dip = ant.get_dipole_for_polarization(polarization)
        return dip.get_cleaned_trace()

    def get_dipole_trace(self,
                         station_name,
                         antenna_id,
                         polarization,
                         kind):
        """
        :param antenna_id: The antenna id ("a"+str(even_dipole_id)), of the form "a028010086".
        :type antenna_id: str
        :param station_name: Name of the station, for example "CS002".
        :type station_name: str
        :param polarization: pol value 0/1
        :type polarization: int
        :param kind: raw/cleaned/calibrated/Ex/Ey/Ez/time.
        :type kind:str
        """
        if kind not in ['raw', 'cleaned', 'calibrated', 'Ex', 'Ey', 'Ez', 'time']:
            logger.error(f"Invalid entry for kind:{kind}")
            return None
        sta = self.get_station_by_name(station_name)
        ant = sta.get_antenna_by_id(antenna_id)
        dip = ant.get_dipole_for_polarization(polarization)
        trace = None
        if kind == "raw":
            trace = dip.get_raw_trace()
        if kind == "cleaned":
            trace = dip.get_cleaned_trace()
        if kind == "calibrated":
            trace = dip.get_calibrated_trace()
        # TODO: If kind='Ex/Ey/Ez/time'. @Nikos. Be careful about time.
        # Caution: what is the meaning of zero on time trace?
        return trace

    def get_dipole_spectrum(self,
                            station_name,
                            antenna_id,
                            polarization,
                            kind):
        """
        :param antenna_id: antenna id ("a"+str(even_dipole_id))
        :type antenna_id: str
        :param polarization: pol value 0/1
        :type polarization: int
        :param kind: raw/cleaned/calibrated/Ex/Ey/Ez.
        :type kind:str
        """
        if kind not in ['raw', 'cleaned', 'calibrated', 'Ex', 'Ey', 'Ez']:
            logger.error(f"Invalid entry for kind:{kind}")
            return None

        trace = self.get_dipole_trace(
            station_name,
            antenna_id,
            polarization,
            kind
        )
        spec = 2 * np.abs(np.fft.rfft(trace))
        return spec
