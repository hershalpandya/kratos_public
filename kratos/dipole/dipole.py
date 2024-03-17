import logging
import os

import numpy as np

logger = logging.getLogger(os.path.basename(__file__))


class Dipole:
    def __init__(self, polarization=None, dipole_id=None, data_dict=None):
        """
        Initialize Dipole Data Members
        :param polarization: 0/1 for LOFAR. or 'x','y','z' (in the future)
        :type polarization: int
        :param dipole_id: '0x000xx' strings in LOFAR
        :type dipole_id: str
        """
        if data_dict:
            self.set_members_from_data_dict(data_dict)
            return

        self.__polarization = polarization
        self.__dipole_id = dipole_id
        self.__raw_trace = np.array([])
        self.__cleaned_trace = np.array([])
        self.__calibrated_trace = np.array([])
        self.__flagged = False
        self.__flagged_reason=""
        return

    def get_polarization(self):
        return self.__polarization

    def get_dipole_id(self):
        return self.__dipole_id

    def is_flagged(self):
        return self.__flagged

    def get_flagged_reason(self):
        return self.__flagged_reason

    def set_flagged(self, flagged_reason=""):
        self.__flagged = True
        self.__flagged_reason = flagged_reason
        return

    def clear_flagged(self):
        self.__flagged = False
        self.__flagged_reason = ""

    def get_raw_trace(self):
        """
        :return: raw_trace in units of ADC counts
        :rtype: numpy.ndarray
        """
        return self.__raw_trace

    def set_raw_trace(self, raw_trace):
        """
        :param raw_trace: raw dipole trace in ADC counts
        :type raw_trace: numpy.ndarray
        """
        if not isinstance(raw_trace, np.ndarray):
            logger.error("Dipole raw_trace has to be a numpy array.")
            raise Exception()

        self.__raw_trace = raw_trace

        return

    def get_cleaned_trace(self):
        """
        :return: cleaned_trace in units of ADC counts
        :rtype: numpy.ndarray
        """
        return self.__cleaned_trace

    def set_cleaned_trace(self, cleaned_trace):
        """
        :param cleaned_trace: rfi cleaned dipole trace in ADC counts
        :type cleaned_trace: numpy.ndarray
        """
        if not isinstance(cleaned_trace, np.ndarray):
            logger.error("Dipole cleaned_trace has to be a numpy array.")
            raise Exception()

        self.__cleaned_trace = cleaned_trace

        return

    def get_calibrated_trace(self):
        """
        :return: calibrated_trace in units of V
        :rtype: numpy.ndarray
        """
        return self.__calibrated_trace

    def set_calibrated_trace(self, calibrated_trace):
        """
        :param calibrated_trace: absolute calibrated_trace in V
        :type calibrated_trace: numpy.ndarray
        """
        if not isinstance(calibrated_trace, np.ndarray):
            logger.error("Dipole calibrated_trace has to be a numpy array.")
            raise TypeError

        self.__calibrated_trace = calibrated_trace

        return

    def get_data_dict_from_members(self):
        """
        Returns a dictionary with keys and values to be saved to disk.

        :return: dictionary with dipole keys and values
        :rtype: dict
        """
        data_dict = {"dipole_id": self.__dipole_id,
                     "polarization": self.__polarization,
                     "raw_trace": self.__raw_trace,
                     "cleaned_trace": self.__cleaned_trace,
                     "calibrated_trace": self.__calibrated_trace
                     }
        return data_dict

    def set_members_from_data_dict(self, data_dict):
        """
        Accepts a dictionary as input and sets values for members of the class.

        :param data_dict: dictionary with dipole keys
        :type data_dict: dict
        """
        self.__dipole_id = data_dict["dipole_id"]
        self.__polarization = data_dict["polarization"]
        self.__raw_trace = data_dict["raw_trace"]
        self.__cleaned_trace = data_dict["cleaned_trace"]
        self.__calibrated_trace = data_dict["calibrated_trace"]
        return
