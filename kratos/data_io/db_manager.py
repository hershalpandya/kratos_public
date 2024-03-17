import copy
import os
import json
import numpy as np
import logging
from .event_dictionary import event_dict as event_dict_template
from kratos.data_io import storage_paths

logger = logging.getLogger(os.path.basename(__file__))


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def a_keys_subset_of_b_keys(a, b):
    """
    :param a: dictionary
    :param b: dictionary
    :returns: True if a.keys() are subset of b.keys()
    """
    alist = list(a.keys())
    blist = list(b.keys())

    mask = [i in blist for i in alist]
    result = all(mask)

    if not result:
        alist = np.array(alist)
        mask = np.array(mask)
        logger.warning("Added key(s):", alist[~mask])
        logger.warning("adding of new keys to dictionary not allowed")

    return result


def match_keys(a, b):
    all_match = True
    tier1_match = a_keys_subset_of_b_keys(a, b)

    all_match = all_match and tier1_match

    if not all_match:
        return all_match

    for key in a.keys():
        if type(a[key]) == dict:
            tier2_match = a_keys_subset_of_b_keys(a[key], b[key])
            all_match = all_match and tier2_match

    return all_match


class DBManager:
    """
    Manages creating, editing, and reading of json file. Each json file is for one event.
    """

    def __init__(self,path=None):
        """ """
        if path==None:
            path_finder = storage_paths.PathFinder()
            self.path = path_finder.get_event_json_files_root_dir()
        else:
            self.path = path
        return

    def get_json_filename(self, event_id):
        """Converts event id integer to filename

        :param event_id: Event id number
        :type event_id: int
        :returns : filename
        :rtype: str
        """
        err = "event_id cannot be None. Pls set it to a valid integer value."
        if not event_id:
            logger.error(err)
            raise ValueError

        fname = self.path + "/" + "%i.json" % event_id
        return fname

    def get_event_dictionary_from_json_file(self, event_id):
        """Reads json file. If it finds any lists in second tier data, then
        converts them to np arrays.

        :param event_id: Event id number
        :type event_id: int
        :returns : dictionary
        :rtype: dict
        """
        fname = self.get_json_filename(event_id)
        with open(fname) as json_file:
            dict_0 = json.load(json_file)

        # assert match_keys(dict_0, event_dict_template)

        for key in dict_0.keys():
            if type(dict_0[key]) == dict:
                for key2 in dict_0[key].keys():
                    if type(dict_0[key][key2]) == list:
                        dict_0[key][key2] = np.array(dict_0[key][key2])
        return dict_0

    @staticmethod
    def get_event_dictionary_empty():
        """Get an empty event dictionary to fill it in."""
        return copy.deepcopy(event_dict_template)

    def write_event_dictionary_to_disk(self, filled_event_dict):
        """Saves event dictionary as json file on disk. Filename taken from
        event_id stored inside the dictionary.

        :param filled_event_dict: dictionary to be saved.
        """

        if not match_keys(filled_event_dict, event_dict_template):
            logger.error(
                "keys of filled_event_dict don't match \
                         template keys."
            )
            raise ValueError

        fname = self.get_json_filename(filled_event_dict["event_id"])

        if os.path.exists(fname):
            logger.warning(f"Replacing an existing file {fname}")

        with open(fname, "w") as fp:
            json.dump(filled_event_dict, fp, indent=4, cls=NumpyEncoder)
        return
