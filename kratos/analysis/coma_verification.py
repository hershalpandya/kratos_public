import os
import glob
import json
import logging
import argparse

import numpy as np

from tqdm import tqdm

from kratos import event
from kratos.physics import power
from kratos.data_io import storage_paths
from kratos.filter import BandpassFilter, HalfHannFilter

# Read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-re", "--replace_existing",
                    type=bool, help="Whether to replace existing NPZ event files",
                    default=False)
args = parser.parse_args()

logger = logging.getLogger(os.path.basename(__file__))

# Get all the specific file loggers to control their behaviour
tbb_logger = logging.getLogger('raw_tbb_IO.py')
rfi_logger = logging.getLogger('find_rfi.py')
lofar_logger = logging.getLogger('lofar_io.py')
event_logger = logging.getLogger('event.py')
station_logger = logging.getLogger('station.py')

# Create handlers
error_handler = logging.StreamHandler()
error_handler.setLevel(logging.ERROR)

info_handler = logging.StreamHandler()
info_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

error_handler.setFormatter(c_format)
info_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.setLevel(logging.DEBUG)

tbb_logger.addHandler(error_handler)
rfi_logger.addHandler(error_handler)
lofar_logger.addHandler(error_handler)
event_logger.addHandler(error_handler)
station_logger.addHandler(error_handler)

# Make sure the messages are not propagated to the root logger (causing duplicate messages)
logger.propagate = False
tbb_logger.propagate = False
rfi_logger.propagate = False
lofar_logger.propagate = False
event_logger.propagate = False
station_logger.propagate = False

path_finder = storage_paths.PathFinder()
path_finder.get_metadata_dir()

coma_dir = '/vol/astro3/lofar/vhecr/lora_triggered/results_with_abs_calibration/'

list_of_events = os.listdir(coma_dir)  # might not all have JSON files generated

my_bandpass_filter = BandpassFilter()  # use default settings for freq and roll width
my_hann_filter = HalfHannFilter()  # use default half percentage

exist_error = 0
json_error = 0
coma_error = 0
load_event_error = 0
mismatch_error = 0
antenna_error = 0
other_error = 0

total_event = 0
total_station = 0

antennaset_dict = {}

gen = tqdm(list_of_events)
for event_id in gen:
    gen.refresh()

    if os.path.exists(f"COMA_KRATOS_comparison_{event_id}.npz") and not args.replace_existing:
        logger.info(f'NPZ file for {event_id} already exists, skipping...')
        exist_error += 1
        continue
    if not os.path.exists(os.path.join(path_finder.get_event_json_files_root_dir(), f'{event_id}.json')):
        logger.error(f'JSON file for event {event_id} not found, skipping...')
        json_error += 1
        continue

    max_ratio = []
    power_ratio = []
    processed_stations = []

    list_of_stations = [os.path.basename(entry).split('-')[2].split('.')[0]
                        for entry in glob.glob(os.path.join(coma_dir, f'{event_id}/calibrated_pulse_block*.npy'))]

    if len(list_of_stations) == 0:
        logger.error(f'COMA does not contain any NPY files, skipping...')
        coma_error += 1
        continue

    logger.info(f'Processing event {event_id}...')

    try:
        this_event = event.Event(event_id=int(event_id), pathfinder=path_finder)

        this_event.load_LOFAR_stations()
        this_event.load_LOFAR_traces()
    except ValueError:
        logger.error(f'Event {event_id} cannot be found in LORA file, skipping...')
        load_event_error += 1
        continue
    except OSError:
        logger.error(f'Antennas do not start at the same time for event {event_id}, skipping...')
        load_event_error += 1
        continue

    antennaset_dict[event_id] = this_event.get_station_by_name(list_of_stations[0]).get_antenna_set()
    total_event += 1

    # Speed up the process by only processing the stations which are saved on COMA
    for station_name in list_of_stations:
        logger.info(f"Processing station {station_name}...")

        # Get Station object from Event
        this_station = this_event.get_station_by_name(station_name)

        if len(this_station.antennas) != 48:
            antenna_error += 1
            logger.warning(f"Station {station_name} does not have 48 antenna's in TBB file!")

        # In the final array, -1 indicates that the antenna was not included
        max_ratio_array = np.ones((2, len(this_station.antennas))) * -1  # sometimes there are less than 48 antennas
        power_ratio_array = np.ones((2, len(this_station.antennas))) * -1

        try:
            this_station.do_rfi_cleaning()

            this_station.apply_to_all_traces(my_bandpass_filter.apply_to_trace,
                                             trace_type='cleaned', save_type='cleaned')
            this_station.apply_to_all_traces(my_hann_filter.apply_to_trace,
                                             trace_type='cleaned', save_type='cleaned')

            this_station.do_galactic_calibration(int(event_id) + 1262304000, 'LOFAR_LBA', cal='old')

            this_station.apply_calibration_delays(trace_type='calibrated', save_type='calibrated')
        except Exception as e:
            logger.error(f'Something went wrong with station {station_name}: {e} \n Skipping this one...')
            other_error += 1
            continue

        # Get calibrated traces for good antennas in the station
        good_antennas = this_station.get_good_antennas()
        calibrated_trace_kratos = np.array([[antenna.dipoles[1].get_calibrated_trace(),
                                             antenna.dipoles[0].get_calibrated_trace()] for antenna in good_antennas])

        # Additional flagging for power (?)
        # calibrated_power_kratos = power(calibrated_trace_kratos)
        # calibrated_power_kratos_max = np.mean(calibrated_power_kratos) + 5.8 * np.std(calibrated_power_kratos)
        # calibrated_power_kratos_min = np.mean(calibrated_power_kratos) - 5.8 * np.std(calibrated_power_kratos)
        #
        # anomalous_antennas_high_power = np.where(calibrated_power_kratos > calibrated_power_kratos_max)[0]
        # anomalous_antennas_low_power = np.where(calibrated_power_kratos < calibrated_power_kratos_min)[0]
        #
        # logger.warning(f"Antenna's {anomalous_antennas_high_power} have a higher power than usual, removing...")
        # logger.warning(f"Antenna's {anomalous_antennas_low_power} have a lower power than usual, removing...")
        #
        # good_antennas_indices = list(range(len(good_antennas)))
        # if len(anomalous_antennas_high_power) > 0:
        #     good_antennas_indices.remove(anomalous_antennas_high_power)
        # if len(anomalous_antennas_low_power) > 0:
        #     good_antennas_indices.remove(anomalous_antennas_low_power)

        # Get calibrated trace from COMA
        calibrated_trace_coma = np.load(
            os.path.join(
                coma_dir, f"{event_id}", f"calibrated_pulse_block-{event_id}-{station_name}.npy"
            )
        )

        # Retain only non-flagged antennas and reshape array to match COMA
        # logger.debug(good_antennas_indices)
        try:
            # Reshape both arrays to (pol, ant, trace)
            calibrated_trace_kratos = np.swapaxes(calibrated_trace_kratos, 0, 1)
            calibrated_trace_coma = calibrated_trace_coma.reshape(calibrated_trace_kratos.shape, order='F')
        except ValueError:
            logger.warning(f"KRATOS and COMA don't have the same number of antenna's for station {station_name}, "
                           f"skipping...")
            mismatch_error += 1
            continue

        logger.debug(calibrated_trace_kratos.shape)
        logger.debug(calibrated_trace_coma.shape)

        # Swap polarizations on COMA
        calibrated_trace_coma_swapped = np.flip(calibrated_trace_coma, 0)

        max_ratio_array[:, this_station.get_good_antenna_indices()] = \
            np.max(calibrated_trace_kratos, axis=-1) / np.max(calibrated_trace_coma_swapped, axis=-1)

        power_ratio_array[:, this_station.get_good_antenna_indices()] = \
            power(calibrated_trace_kratos) / power(calibrated_trace_coma_swapped)

        max_ratio.append(max_ratio_array)
        power_ratio.append(power_ratio_array)
        processed_stations.append(station_name)

        total_station += 1

    savez_dict = dict()
    for ind, station in enumerate(processed_stations):
        savez_dict[f'{station}_max_ratio'] = max_ratio[ind]
        savez_dict[f'{station}_power_ratio'] = power_ratio[ind]

    np.savez(f"COMA_KRATOS_comparison_{event_id}.npz", **savez_dict)

    r = json.dumps(antennaset_dict)
    with open(f'COMA_verification_antennasets.log', 'w') as json_file:
        json_file.write(r)

    with open(f"COMA_verification.log", "w") as file:
        file.write(f"There were {total_event} events processed \n")
        file.write(f"There were {total_station} stations processed \n \n")

        file.write(f"There were {exist_error + json_error + coma_error + load_event_error} event related errors \n")
        file.write(f" -- There were {exist_error} events which already had an NPZ file \n")
        file.write(f" -- There were {json_error} events for which no JSON file was found \n")
        file.write(f" -- There were {coma_error} events for which there were no NPY files on COMA \n")
        file.write(f" -- There were {load_event_error} events which failed on load \n")

        file.write(f"There were {mismatch_error + antenna_error + other_error} station related errors \n")
        file.write(f" -- There were {mismatch_error} stations which had a different number of antennas on COMA \n")
        file.write(f" -- There were {antenna_error} stations which had less than 48 antennas read in from TBB files \n")
        file.write(f" -- There were {other_error} stations for which another error occurred \n")
