import numpy as np

event_dict = {
    "event_id": None,
    "date_time": None,
    "best_core_x_m": None,
    "best_core_y_m": None,
    "best_zenith_rad": None,
    "best_azimuth_rad": None,
    "best_energy_GeV": None,
    "LORA": {},
    "LOFAR": {},
    "status": {},
}

###############
# LORA
###############

lora_vars = [
    "ROOT_file_name",
    "utc_time_stamp",
    "time_stamp_ns",
    "n_stations",
    "trigger_mode",
    "trigger_condition",
    "zenith_rad",
    "zenith_err_rad",
    "azimuth_rad",
    "azimuth_err_rad",
    "energy_GeV",
    "energy_err_GeV",
    "n_detectors",
    "core_x_m",
    "core_y_m",
    "core_err_x_m",
    "core_err_y_m",
]
for var in lora_vars:
    event_dict["LORA"][var] = None

###############
# LOFAR
###############

lofar_vars = ["TBB_file_names"]

for arr in lofar_vars:
    event_dict["LOFAR"][arr] = np.array([])


###############
# DB Flags
###############
event_dict["status"]["CR_Found"] = False
