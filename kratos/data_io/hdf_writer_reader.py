import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(os.path.basename(__file__))


class HDFWriterReader:
    """
    Interface for writing and reading HDF5 Files.

    Be really careful about backwards compatibility when editing this code.

    ..collapse:: HDF File Structure
        | event
        | ├── event_vars
        | ├── ...
        | ├── LORA (Particle Data Group)
        |   ├── lora_vars
        |   ├── ...
        |   ├── station_1
        |       ├── station_vars
        |       └── ...
        |   └── ... other stations.
        | ├── LOFAR (Radio Data Group)
        |   ├── lofar_vars
        |   ├── ...
        |   ├── station_1
        |       ├── station_vars
        |       ├── ...
        |       ├── antenna_1
        |           ├── antenna_vars
        |           ├── ...
        |           ├── dipole_1
        |               ├── dipole_vars
        |               └── ...
        |           └── ...other dipoles
        |       └── ... other antennas
        |   └── ... other stations

    ..todo::
        * Possibly convert this class into separate functions...
        * ... if __init__() is still empty in the final form.

    """

    def __init__(self):
        """
        Defining all variable lists.
        [, , ,] = variable name followed by dtype and bool for compression.
        """

        # Define String dtype for h5py
        special_str_dtype = h5py.special_dtype(vlen=str)

        self.event_json_vars = [
            ["event_id", "int", False],
            ["date_time", special_str_dtype, False],
            ["best_core_x_m", "float", False],
            ["best_core_y_m", "float", False],
            ["best_zenith_rad", "float", False],
            ["best_azimuth_rad", "float", False],
            ["best_energy_GeV", "float", False],
        ]

        self.LORA_json_vars = [
            ["ROOT_file_name", special_str_dtype, False],
            ["utc_time_stamp", "int", False],
            ["time_stamp_ns", "int", False],
            ["n_stations", "int", False],
            ["trigger_mode", "int", False],
            ["trigger_condition", "int", False],
            ["zenith_rad", "float", False],
            ["zenith_err_rad", "float", False],
            ["azimuth_rad", "float", False],
            ["azimuth_err_rad", "float", False],
            ["energy_GeV", "float", False],
            ["energy_err_GeV", "float", False],
            ["n_detectors", "int", False],
            ["core_x_m", "float", False],
            ["core_y_m", "float", False],
            ["core_err_x_m", "float", False],
            ["core_err_y_m", "float", False],
        ]

        self.station_vars = [
            ["data_filename", special_str_dtype, False],
            ["station_name", special_str_dtype, False],
            ["antennaset", special_str_dtype, False],
            ["antenna_model", special_str_dtype, False],
            ["clock_frequency_MHz", "float", False],
            ["dirty_channels_blocksize", "float", False],
            ["trace_length_nbins", "int", False],
            ["dirty_channels", "int", True],
            ["time_axis_ns", "float", True],
            ["raw_avg_powerspectrum", "float", True],
            ["cleaned_avg_powerspectrum", "float", True],
        ]

        self.antenna_vars = [
            ["position_m", "float", False],
            ["E_trace", "float", True],
            ["antenna_id", special_str_dtype, False],
        ]

        self.dipole_vars = [
            ["dipole_id", special_str_dtype, False],
            ["polarization", "int", False],
            ["raw_trace", "float", True],
            ["cleaned_trace", "float", True],
            ["calibrated_trace", "float", True],
        ]

        return

    @staticmethod
    def h5py_read_datasets_helper(h5py_dataset, dtype, compression):
        """
        Manages idiosyncrasies of loading from hdf5 using h5py.

        :param h5py_dataset: reference to dataset in hdf5 file.
        :type h5py_dataset: h5py.dataset()
        :param dtype: data type
        :type dtype: str / h5py.special_dtype(vlen=str)
        :param compression: an array is compressed, a scalar is not.
        :type compression: bool
        """
        special_str_dtype = h5py.special_dtype(vlen=str)
        if dtype == special_str_dtype:
            h5py_dataset = h5py_dataset.asstr()
        x = h5py_dataset[()]
        # if its not compressed, its a scalar. (as per definition in __init__)
        if not compression:
            # hdf5 method of storing None is empty array.
            # make it a None again.
            if isinstance(x, np.ndarray) and len(x) == 0:
                x = None
        return x

    @staticmethod
    def h5py_write_datasets_helper(group, dictionary, vars_list):
        """
        Helper function for improving readability of Write_Event_To_HDF5()

        :param group: group object to which datasets will be added
        :type group: h5py group object
        :param dictionary: contains keys and values of vars in vars_list
        :type dictionary: dict
        :param vars_list: N_vars long list with each element being:
        ['var_name',dtype,True/False].
        dtype is a string for 'float' or 'int' or special_str_dtype
        True/False is for whether to use compression or not
        :type vars_list: list of shape (N_vars, 3)

        :return: group with datasets added
        """

        for var, dtype, compression in vars_list:
            if var not in dictionary.keys():
                logger.error(f"{var} not found in dictionary keys.")
                raise ValueError
                
            value = dictionary[var]
            if not isinstance(value, np.ndarray):
                if value==None:
                    value = []  # HDF5 equivalent to None
                    
            if compression:
                group.create_dataset(
                    var, data=value, dtype=dtype, compression="gzip"
                )
            else:
                group.create_dataset(var, data=value, dtype=dtype)

        return group

    @staticmethod
    def __are_equal(x, y):
        """
        Following datatypes supported:
        np.ndarray, float, int, str, np.float64, np.int64, type(None).

        :param x: first variable
        :param y: second variable
        :return: True if x==y
        :rtype: bool
        """
        if type(x) == list: x = np.array(x)
        if type(y) == list: y = np.array(y)
        
        if type(x) == np.ndarray:
            if x.dtype != object and y.dtype!=object:
                if np.isnan(x).all() and np.isnan(y).all():
                    return True
            return np.array_equal(x, y)
        
        elif type(x) in [float, int, str, np.float64, np.int64, type(None)]:
            return x == y
        
        else:
            logger.warning("found not-implemented type:", type(x))
            return False
        
    @staticmethod
    def __compare_dictionaries(d1, d2):
        """
        Compares if two nested dictionaries are identical and equal.

        :param d1: first dictionary
        :type d1: dict
        :param d2: second dictionary
        :type d2: dict
        :return: True if d1==d2.
        :rtype: bool
        """
        equal = True

        if d1.keys() != d2.keys():
            logger.warning(f"d1.keys() = {d1.keys()}")
            logger.warning(f"d2.keys() = {d2.keys()}")
            logger.warning("Not EQUAL")
            equal = False

        for key in d1.keys():
            if not equal:
                break

            if type(d1[key]) == dict:
                equal = HDFWriterReader.__compare_dictionaries(d1[key], d2[key])
            else:
                equal = HDFWriterReader.__are_equal(d1[key], d2[key])
                if not equal:
                    print(key, d1[key], d2[key])
                    logger.warning("Not EQUAL")
        return equal

    def write_event_dictionary_to_hdf(self, data_dict, fname):
        """
        * Writes the Event Dictionary into a HDF5 file.

        :param data_dict: Event level dictionary
        :type data_dict: dict
        :param fname: full path to HDF5 file
        :type fname: str
        """

        if os.path.exists(fname):
            logger.warning(f"File already exists. Overwriting. {fname}")

        # Initiate file instance
        f = h5py.File(fname, "w")

        # Populate event group
        group_event = f.create_group("event")

        # event vars from json file / event_dictionary
        group_event = self.h5py_write_datasets_helper(
            group_event, data_dict, self.event_json_vars
        )

        # Populate event/LORA group
        group_LORA = group_event.create_group("LORA")

        # LORA vars from json file / event_dictionary
        group_LORA = self.h5py_write_datasets_helper(
            group_LORA, data_dict["LORA"], self.LORA_json_vars
        )

        # Populate event/LOFAR group
        group_LOFAR = group_event.create_group("LOFAR")

        # loop over all stations.
        station_keys = [i for i in data_dict["LOFAR"].keys() if "Station_" in i]

        for key_sta in station_keys:
            group_station = group_LOFAR.create_group(key_sta)

            group_station = self.h5py_write_datasets_helper(
                group_station, data_dict["LOFAR"][key_sta], self.station_vars
            )

            # loop over all antennas
            antenna_keys = [
                i for i in data_dict["LOFAR"][key_sta].keys() if "Antenna_" in i
            ]

            for key_ant in antenna_keys:
                group_antenna = group_station.create_group(key_ant)

                group_antenna = self.h5py_write_datasets_helper(
                    group_antenna,
                    data_dict["LOFAR"][key_sta][key_ant],
                    self.antenna_vars,
                )

                # loop over all dipoles.
                dipole_keys = [
                    i
                    for i in data_dict["LOFAR"][key_sta][key_ant].keys()
                    if "Dipole_" in i
                ]

                for key_dip in dipole_keys:
                    group_dipole = group_antenna.create_group(key_dip)

                    group_dipole = self.h5py_write_datasets_helper(
                        group_dipole,
                        data_dict["LOFAR"][key_sta][key_ant][key_dip],
                        self.dipole_vars,
                    )
        f.close()

        # sanity check
        self.compare_dictionary_with_hdf(data_dict,fname)
        return

    def read_hdf_to_dictionary(self, fname):
        """
        * Reads the data from given HDF5 into the event object.
        * Also populates Station/Antenna/Dipole objects.

        :param fname: full path to HDF5 file
        :type fname: str
        :return: dictionary with all event data.
        :rtype: dict
        """
        data_dict = {}

        # open file
        f = h5py.File(fname, "r")

        # load event level info:
        for key, dtype, compression in self.event_json_vars:
            data_dict[key] = self.h5py_read_datasets_helper(
                f["event"][key], dtype, compression
            )

        # load lora info:
        data_dict["LORA"] = {}
        for key, dtype, compression in self.LORA_json_vars:
            data_dict["LORA"][key] = self.h5py_read_datasets_helper(
                f["event"]["LORA"][key], dtype, compression
            )

        # load LOFAR stations:
        data_dict["LOFAR"] = {}
        station_keys = [i for i in f["event"]["LOFAR"].keys() if "Station_" in i]
        for key_sta in station_keys:
            data_dict["LOFAR"][key_sta] = {}
            for key, dtype, compression in self.station_vars:
                data_dict["LOFAR"][key_sta][key] = self.h5py_read_datasets_helper(
                    f["event"]["LOFAR"][key_sta][key], dtype, compression
                )

            # loop over all antennas.
            antenna_keys = [
                i for i in f["event"]["LOFAR"][key_sta].keys() if "Antenna_" in i
            ]

            for key_ant in antenna_keys:
                data_dict["LOFAR"][key_sta][key_ant] = {}
                for key, dtype, compression in self.antenna_vars:
                    data_dict["LOFAR"][key_sta][key_ant][
                        key
                    ] = self.h5py_read_datasets_helper(
                        f["event"]["LOFAR"][key_sta][key_ant][key], dtype, compression
                    )

                # loop over all dipoles.
                dipole_keys = [
                    i
                    for i in f["event"]["LOFAR"][key_sta][key_ant].keys()
                    if "Dipole_" in i
                ]

                for key_dip in dipole_keys:
                    data_dict["LOFAR"][key_sta][key_ant][key_dip] = {}
                    for key, dtype, compression in self.dipole_vars:
                        data_dict["LOFAR"][key_sta][key_ant][key_dip][
                            key
                        ] = self.h5py_read_datasets_helper(
                            f["event"]["LOFAR"][key_sta][key_ant][key_dip][key],
                            dtype,
                            compression,
                        )

        f.close()

        return data_dict

    def compare_dictionary_with_hdf(self,data_dict,fname):
        """
        Compares whether dictionary in code memory (which can be obtained
        by doing ```this_event.get_data_dict_from_members()```) is same as the
        written hdf5 file.

        This utility has been developed because writer and reader of HDFWriter
        are two different methods. If in the future, you edit writer to include
        or exclude something, the reader should be appropriately edited.

        This method should be called whenever a file is written. (To make sure
        same file is readable as well).

        Be really careful about backwards compatibility when editing this code.

        :param data_dict: kratos event dictionary
        :type data_dict: dict
        :param fname: full path to hdf5 file
        :type fname: str
        """
        logger.info("Checking that written file is readable...")

        d_read_in = self.read_hdf_to_dictionary(fname)
        equal = self.__compare_dictionaries(data_dict,d_read_in)
        
        if not equal:
            logger.warning("Read/Write methods of HDFWriter"
                            " do not produce same result.")
            raise ValueError
        return

