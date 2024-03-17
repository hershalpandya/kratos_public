import logging
import os

import numpy as np

from kratos.dipole import Dipole

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


class Antenna:
    def __init__(self, two_dipole_ids=None,
                 position_m=np.array([]),
                 data_dict=None
                 ):
        """
        :param two_dipole_ids: names/ids for two dipoles of this antenna.
        :type two_dipole_ids: (str, str) or (int, int)
        :param: position_m: Position is assigned by Station() while creating Antenna objects.
        :type position_m: np.ndarray
        :return: None
        """

        if data_dict:
            self.set_members_from_data_dict(data_dict)
            return

        self.dipoles: list[Dipole] = []

        self.__E_trace_V_m = np.array([])  # in V/m, in xyz LOFAR coordinate system
        # TODO: document the reference for zero of time trace @Katie
        self.__position_m = None
        self.__antenna_id = None
        self.__flagged = False
        self.__flagged_reason = ""
        # TODO: Get() Set() for time. @Nikos

        # X, Y, Z in meters from array center
        if not isinstance(position_m, np.ndarray):
            if not len(position_m) == 3:
                logger.error("Position required to be a 3 sized numpy array")
        self.__position_m = position_m

        # TODO: make polarization flexible?
        if two_dipole_ids:
            if len(two_dipole_ids) != 2:
                logger.error(f"wrong shape for two_dipole_ids:{two_dipole_ids}")
                raise ValueError

            self.dipoles.append(
                Dipole(dipole_id=str(two_dipole_ids[0]), polarization=0)
            )

            self.dipoles.append(
                Dipole(dipole_id=str(two_dipole_ids[1]), polarization=1)
            )

            # Assign antenna_id
            # antenna_id = "a" + str(dipole_id) of even numbered dipole_id
            two_dipole_ids = np.array(two_dipole_ids, dtype=str)
            even_or_odd = np.array([int(a) % 2 for a in two_dipole_ids])
            if len(np.unique(even_or_odd)) != 2:
                logger.error(f"Strictly one of the dipole ids provided:{two_dipole_ids}\
                needs to be even for unique nomenclature of antenna_id")
                raise ValueError

            self.__antenna_id = "a" + str(two_dipole_ids[even_or_odd == 0][0])
        return

    def get_position(self):
        """
        :return: position in meters [x,y,z]
        :rtype: np.ndarray
        """
        return self.__position_m

    def get_antenna_id(self):
        """
        :return: antenna_index = dipole_id%100
        :rtype: int
        """
        return self.__antenna_id

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

    def get_dipole_for_polarization(self, polarization):
        """
        Get the dipole with polarization ``polarization``. If no antenna has the requested polarization, the function
        returns None.

        :param polarization: polarization (0/1/2)
        :type polarization: int
        :return: Dipole with the requested polarization or None if it cannot be found.
        :rtype: Dipole() Obj.
        """

        for dipole in self.dipoles:
            if dipole.get_polarization() == polarization:
                return dipole

        logger.error(f"Failed to find dipole with polarization {polarization}.")

        return None

    def get_polarizations(self):
        polarizations = []
        for dipole in self.dipoles:
            polarizations.append(dipole.get_polarization())
        return polarizations

    def get_data_dict_from_members(self):
        """
        Returns a dictionary with keys and values that have to be saved to disk

        :return: nested dictionary with antenna keys and values
        :rtype: dict
        """
        data_dict = {"antenna_id": self.__antenna_id,
                     "position_m": self.__position_m,
                     "E_trace": self.__E_trace,
                     }

        for i, dip in enumerate(self.dipoles):
            data_dict[f"Dipole_{i}"] = dip.get_data_dict_from_members()

        return data_dict

    def set_members_from_data_dict(self, data_dict):
        """
        Accepts a dictionary as input and sets values for members of the class.

        :param data_dict: nested dictionary with antenna keys and dipole keys
        :type data_dict: dict
        """
        self.__antenna_id = data_dict["antenna_id"]
        self.__position_m = data_dict["position_m"]
        self.__E_trace = data_dict["E_trace"]

        self.dipoles = []

        dipole_keys = [i for i in list(data_dict.keys()) if 'Dipole_' in i]

        for key in dipole_keys:
            new_dipole = Dipole(data_dict=data_dict[key])
            self.dipoles.append(new_dipole)
        return

    def trim_traces(self, new_tracelength_nbins):
        """
        Trims all traces. Time_array, Raw, Clean, Calibrated, Ex,Ey, and Ez.

        Trims traces for all stations to new value.
        Space saving technique. To be called after RFI cleaning.

        :param new_tracelength_nbins: trace length to store in h5 files.
        :type new_tracelength_nbins: int
        """
        # TODO: implement this.
        # first trim Time_array, Ex, Ey, Ez which will be property of antenna.
        # then loop over all dipoles and trim their raw, clean, calibrated.
        return

    # TODO: E_trace may be better and smarter. It is disconnected to dipoles for now.
    def get_E_trace(self):
        """
        :return: E_trace in units of V/m, shaped as (xyz, samples).
        :rtype: numpy.ndarray
        """
        return self.__E_trace_V_m

    def set_E_trace(self, trace):
        """
        :param trace: Electric field vector in V/m, shaped as (xyz, samples).
        :type trace: numpy.ndarray
        """
        if not isinstance(trace, np.ndarray):
            logger.error("Antenna Ex_trace has to be a numpy array.")
            raise TypeError

        self.__E_trace_V_m = trace

        return
