from datetime import datetime

import numpy as np

from kratos.data_io import raw_tbb_IO, raw_tbb_io_metadata
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


class GetLOFARTraces:
    def __init__(
        self, tbb_h5_filename, metadata_dir, time_s, time_ns, trace_length_nbins
    ):
        """
        A Class to facilitate getting traces from LOFAR TBB HDF5 Files

        :param time_s: event trigger timestamp in UTC seconds
        :type time_s: int
        :param time_ns: event trigger timestamp in ns past UTC second
        :type time_ns: int
        :param trace_length_nbins: desired length of trace to be loaded from \
        TBB HDF5 files. Does not affect trace size read-in for RFI cleaning
        :type trace_length_nbins: int

        """
        self.metadata_dir = metadata_dir
        self.data_filename = tbb_h5_filename
        self.trace_length_nbins = trace_length_nbins
        self.block_number = None
        self.sample_number_in_block = None
        self.tbb_file = None
        self.time_s = time_s
        self.time_ns = time_ns
        self.alignment_shift = None

        self.setup_trace_loading()

    def setup_trace_loading(self):
        """
        Opens the file and sets some variables.
        so that get_trace() can be called repeatedly for different dipoles.
        """
        self.tbb_file = open_TBB(self.data_filename, metadata_dir=self.metadata_dir)
        sample_number = self.tbb_file.get_nominal_sample_number()
        timestamp = self.tbb_file.get_timestamp()
        station_clock_offsets = raw_tbb_io_metadata.getClockCorrections(
            metadata_dir=self.metadata_dir
        )
        this_station_name = self.tbb_file.get_station_name()

        logger.info("Getting clock offset for station %s" % this_station_name)
        this_clock_offset_ns = 1.0e9 * station_clock_offsets[this_station_name]
        logger.info("Clock offset is %1.4e ns" % this_clock_offset_ns)

        packet = lora_timestamp_to_blocknumber(
            self.time_s,
            self.time_ns,
            timestamp,
            sample_number,
            clockoffset_ns=this_clock_offset_ns,
            blocksize=self.trace_length_nbins,
        )

        self.block_number, self.sample_number_in_block = packet

        self.alignment_shift = -(
            self.trace_length_nbins // 2 - self.sample_number_in_block
        )  # minus sign, apparently...

        logger.info(
            "Block number = %d, sample number in block = %d, alignment shift = %d"
            % (self.block_number, self.sample_number_in_block, self.alignment_shift)
        )

    def check_trace_quality(self):
        """
        Check all traces recorded from the TBB against quality requirements.
        """
        dipole_names = np.array(self.tbb_file.get_antenna_names())

        # Find the dipoles whose starting sample number and/or number of samples recorded deviates from the median
        sample_number_per_antenna = self.tbb_file.get_all_sample_numbers()
        data_length_per_antenna = self.tbb_file.get_full_data_lengths()

        median_sample_number = np.median(sample_number_per_antenna)
        median_data_length = np.median(data_length_per_antenna)

        deviating_dipole_sample_number = np.where(
            np.abs(
                sample_number_per_antenna - median_sample_number
            ) > median_data_length / 4
        )[0]

        deviating_dipole_starting_later = np.where(
            sample_number_per_antenna > median_sample_number
        )[0]

        deviating_dipole_data_length = np.where(
            np.abs(
                data_length_per_antenna - median_data_length
            ) > median_data_length / 10
        )[0]

        deviating_dipoles = np.unique(
            np.concatenate(
                (
                    deviating_dipole_sample_number,
                    deviating_dipole_starting_later,
                    deviating_dipole_data_length
                )
            )
        )

        # Also check if some dipoles are missing their counterpart
        all_dipoles = [int(x) % 100 for x in self.tbb_file.get_antenna_names()]
        dipoles_missing_counterpart = [x for x in all_dipoles if (x + (1 - 2 * (x % 2))) not in all_dipoles]

        # Use sets for superior search performance
        # Index with lists to make it work for empty arrays
        return set(dipole_names[list(deviating_dipoles)]), set(dipole_names[dipoles_missing_counterpart])

    def get_trace(self, dipole_id):
        """
        :param dipole_id: dipole id string
        :type dipole_id: str
        """

        start_sample = self.trace_length_nbins * self.block_number
        start_sample += self.alignment_shift

        trace = self.tbb_file.get_data(
            start_sample, self.trace_length_nbins, antenna_ID=dipole_id
        )

        return trace

    def close_file(self):
        self.tbb_file.close_file()
        return


def tbb_filetag_from_utc(timestamp):
    """Returns TBB filename based on UTC timestamp

    :param timestamp: UTC timestamp from GPS
    :type timestamp: int
    :return: Filename
    :rtype: str
    """
    # utc_timestamp_base = 1262304000  # Unix timestamp on Jan 1, 2010 (date -u --date "Jan 1, 2010 00:00:00" +"%s")

    dt_object = datetime.utcfromtimestamp(timestamp)
    year = dt_object.year
    month = dt_object.month
    day = dt_object.day
    hour = dt_object.hour
    minute = dt_object.minute
    sec = dt_object.second
    radio_file_tag = "D" + str(year) + str(month).zfill(2) + str(day).zfill(2)
    radio_file_tag += "T" + str(hour).zfill(2) + str(minute).zfill(2)
    radio_file_tag += str(sec).zfill(2)

    return radio_file_tag


def event_id_from_utc(timestamp):
    """Returns Event Id from UTC timestamp.
    Basically subtracts the huge number for 1st Jan 2000 or some such
    date.

    :param timestamp: UTC timestamp from GPS
    :type timestamp: int
    :return: event_id
    :rtype: int
    """
    event_id_offset = 1262304000
    return timestamp - event_id_offset


def lora_timestamp_to_blocknumber(
    lora_seconds,
    lora_nanoseconds,
    starttime,
    samplenumber,
    clockoffset_ns=1e4,
    blocksize=2**16,
    samplingfrequency=200,
):
    """Calculates block number corresponding to LORA timestamp and the
    sample number within that block (i.e. returns a tuple
    (``blocknumber``,``samplenumber``)).
    Input parameters:
    =================== ==============================
    *lora_seconds*      LORA timestamp in seconds (UTC timestamp, second after 1st January 1970).
    *lora_nanoseconds*  LORA timestamp in nanoseconds.
    *starttime*         LOFAR_TBB timestamp.
    *samplenumber*      Sample number.
    *clockoffset*       Clock offset between LORA and LOFAR.
    *blocksize*         Blocksize of the LOFAR data.
    =================== ==============================
    """

    lora_samplenumber = (
        (lora_nanoseconds - clockoffset_ns) * samplingfrequency * 1e-3
    )  # MHz to nanoseconds

    value = (lora_samplenumber - samplenumber) + (
        lora_seconds - starttime
    ) * samplingfrequency * 1e6

    if value < 0:
        raise ValueError("Event not in file.")

    return int(value / blocksize), int(value % blocksize)


def get_metadata(filename, metadata_dir):
    """
    Get metadata from TBB file.

    :param filename: Path of the TBB file.
    :type filename: str or Path
    :param metadata_dir: Path to the TBB metadata directory.
    :type metadata_dir: str or Path
    :return:
    """
    logger.info("Getting metadata from filename: %s" % filename)
    tbb_file = raw_tbb_IO.MultiFile_Dal1(filename, metadata_dir=metadata_dir)
    station_name = tbb_file.get_station_name()
    antennaset = tbb_file.get_antenna_set()
    # TODO: fix naming compliance. But for that, will have to change \
    #  Brian's code which will break future updates... you have been warned.
    clock_frequency = tbb_file.SampleFrequency

    ns_per_sample = 1.0e9 / clock_frequency
    logger.info("The file contains %3.2f ns per sample" % ns_per_sample)  # test
    time_s = tbb_file.get_timestamp()
    time_ns = ns_per_sample * tbb_file.get_nominal_sample_number()

    positions = tbb_file.get_LOFAR_centered_positions()
    dipole_ids = tbb_file.get_antenna_names()
    
    # Try to extract calibration delays from TBB metadata
    calibration_delays = tbb_file.get_timing_callibration_delays(force_file_delays=True)

    tbb_file.close_file()

    return (
        station_name,
        antennaset,
        time_s,
        time_ns,
        clock_frequency,
        positions,
        dipole_ids,
        calibration_delays,
    )  # switch to dict? But have to choose keys and read the m out again...


def open_TBB(filename, metadata_dir):
    return raw_tbb_IO.MultiFile_Dal1(filename, metadata_dir)
