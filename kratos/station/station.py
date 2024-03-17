import logging
import os

import numpy as np

from .. import physics
from ..antenna import Antenna
from ..data_io import find_rfi
from ..data_io import lofar_io
from ..physics import unfold_antenna_response
from ..physics import calc_interferometer_phase, phase_to_complex

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


def decode_sql_output(sql_output):
    import base64
    import pickle

    output_list = []
    for row in sql_output:
        newRow = []
        for item in row:
            if type(item) == str and item.startswith("base64_"):
                # import pdb; pdb.set_trace()
                decoded_string = base64.b64decode(item[len("base64_"):].encode('latin1'))
                # newRow.append(pickle.loads(item[len("base64_"):].decode("base64")))
                newRow.append(pickle.loads(decoded_string, encoding='latin1'))  # , encoding='bytes'))
            else:
                newRow.append(item)
        output_list.append(newRow)
    return output_list


class Station:
    def __init__(self, data_filename=None, data_dict=None, metadata_dir=None):
        """
        Initialize Station() Members to ``None``

        :param data_filename: full path to TBB HDF5 File For This Station
        :type data_filename: [str]
        :param data_dict: dictionary with Station and sublevel keys.
        :type data_dict: dict
        :param metadata_dir: path to metadata. generally kratos_src/data.
        :type metadata_dir: str
        """

        if not metadata_dir:
            logger.error("Need a metadata_dir to pass to raw_tbb_io.")
            raise Exception

        self.__metadata_dir = metadata_dir

        if data_dict:
            self.set_members_from_data_dict(data_dict)
            return

        self.antennas: list[Antenna] = []

        self.__data_filename = data_filename
        self.__station_name = None
        self.__antenna_set = None
        self.__clock_frequency_MHz = None
        self.__dirty_channels = np.array([])
        self.__dirty_channels_blocksize = None
        self.__avg_powerspectrum = np.array([])
        self.__avg_antenna_power = np.array([])
        self.__antenna_model = None  # string
        self.__trace_length_nbins = None
        self.__time_axis_ns = np.array([])
        self.__positions = np.array([])
        self.__calibration_delays = np.array([])

        # TODO: load time_axis from TBB file - one per two dipoles.@Katie
        # TODO: setters and getters for all these? :-p @Arthur
        return

    def get_station_name(self):
        return self.__station_name

    def get_antenna_set(self):
        return self.__antenna_set

    def get_clock_frequency(self):
        return self.__clock_frequency_MHz

    def get_trace_length_nbins(self):
        return self.__trace_length_nbins

    def set_trace_length_nbins(self, trace_length):
        self.__trace_length_nbins = trace_length
        return

    def apply_to_all_traces(self, func, *args, trace_type='raw', save_type=None, vectorise=True, **kwargs):
        """
        Apply a function to all the traces of a given type in the station. Additional arguments to the function
        can be provided through the args and kwargs, so this method executes `func(trace, *args, **kwargs)` for
        every `trace` in the station.

        If the parameter `save_type` is specified, the results will be saved in the traces of the corresponding
        type of the Dipole instances present in the Station instance. The method also returns the result of this
        operation as a Numpy array.

        By default, the function `func` will be considered vectorised, i.e. it will be applied to the packed antenna
        traces array directly. The function must accept a keyword argument called `axis` to indicate the axis over
        which it should be applied. This will be set equal the dimension along which the traces are stored.

        :param func: The function to apply to each trace.
        :type func: function
        :param trace_type: The type of trace to load. Must be "raw", "cleaned" or "calibrated", defaults to "raw".
        :type trace_type: str
        :param save_type: If specified, the result is saved in this trace type.
        :type save_type: str, optional
        :param vectorise: Whether to handle `func` as a vectorised function. If True (the default), the function must
        accept `axis` as a keyword parameter.
        :type vectorise: bool
        :return: The resulting traces in an array shaped as (polarisations, antennas, N), where N is the length of
        the traces after applying `func` to them.
        """
        # Collect all the traces in array with shape (pol, ant, trace)
        all_traces = self.__pack_antenna_traces(trace_type)

        # Apply the function to every trace (eg. dim=-1)
        if vectorise:
            # If the function is vectorised, applying it directly is faster than np.apply_along_axis
            applied_traces = func(all_traces, axis=-1, *args, **kwargs)
        else:
            applied_traces = np.apply_along_axis(func, -1, all_traces, *args, **kwargs)

        # If required, store the handled traces back into variables
        if save_type is not None:
            self.__unpack_antenna_traces(applied_traces, save_type)

        return applied_traces

    def load_LOFAR_antennas(self):  # do this with filename input, given from Event?
        """
        Read in LOFAR TBB file, to initiate Antenna() and Dipole() Objs
        based on the number of antenna in the file.
        """
        # may want to have try/except later
        logger.info("data filename = %s" % self.__data_filename)

        station_metadata = lofar_io.get_metadata(self.__data_filename,
                                                 self.__metadata_dir)

        (
            self.__station_name,
            self.__antenna_set,
            tbb_file_time_s,
            tbb_file_time_ns,
            self.__clock_frequency_MHz,
            self.__positions,
            dipole_ids,
            calibration_delays
        ) = station_metadata

        # lofar_io.get_metadata returns clock frequency in Hz
        self.__clock_frequency_MHz /= 1e6

        # reshape cable delays to be consistent with dipole array shape, ie. (2, ant)
        # -> self.__calibration_delays[0] are the delays for the dipoles with polarization = 0
        self.__calibration_delays = np.reshape(calibration_delays, (2, -1), order='F')

        # Data reader returns only antennas with 2 valid dipoles in list
        # positions is an array of type :
        # [ [x,y,z]_pol0_dipole0, [x,y,z]_pol1_dipole0,
        #                         [x,y,z]_pol0_dipole1, ....]
        # len(positions)= nof_dipoles

        for i in range(0, len(dipole_ids), 2):
            this_position = np.array(self.__positions[i])
            if np.shape(this_position) != (3,):
                logger.error("Position read from metadata invalid.")
                raise ValueError

            two_dipole_ids = dipole_ids[i:i + 2]
            if not isinstance(two_dipole_ids[0], str):
                logger.error("Dipole id is expected to be a string.")
                raise ValueError

            # LOFAR does not have an antenna id. Only Dipole Ids.
            # assigning a simple sequential antenna id. to not confuse.
            # whenever required, always ask dipoles what their id is.
            new_antenna = Antenna(two_dipole_ids=two_dipole_ids,
                                  position_m=this_position)
            self.antennas.append(new_antenna)
        return

    def get_time_ns_array_for_trace(self):
        """
        ..TODO: This is incorrect implementation. In this implementation, it\
        will return arbitrary time axis. The zero of time axis has no meaning.
        The time axis needs to be fixed in :py:func: `kratos.antenna.Antenna.__time_axis_ns` variable
        using appropriate function. And then a getter has to be defined to give access
        to it.

        Get time axis array for this station's traces.
        Uses its own known clock_frequency_MHz and trace_length_nbins.
        """
        x = physics.get_time_ns_array_for_trace(self.__clock_frequency_MHz,
                                                self.__trace_length_nbins)
        return x

    def get_antenna_by_id(self, antenna_id):
        """
        returns antenna pertaining to given antenna id
        antenna_id = "a"+str(dipole_id) for even numbered dipole.
        """
        for ant in self.antennas:
            if ant.get_antenna_id() == antenna_id:
                return ant

        logger.error(f"Antenna for id: {antenna_id} not found.")
        return None

    def get_good_antennas(self):
        return [antenna for antenna in self.antennas if not antenna.is_flagged()]

    def get_good_antenna_indices(self):
        return [not (antenna.is_flagged()) for antenna in self.antennas]

    def get_fftfreq_MHz(self):
        """
        Get fftfreq MHz array for this station.
        Uses its own known clock_frequency_MHz and trace_length_nbins.
        """
        x = physics.get_fftfreq_MHz(self.__clock_frequency_MHz,
                                    self.__trace_length_nbins)
        return x

    def load_LOFAR_traces(self,
                          LORA_trigger_time_s,
                          LORA_trigger_time_ns,
                          trace_length_nbins=65536):
        # takes self.__data_filename
        """
        Reads traces from H5 file, \
        assigns to relevant Station.Antenna.Dipole.set_raw_trace().

        :param LORA_trigger_time_s: the time in seconds of the trigger, used to locate the pulse in the signals.
        :param LORA_trigger_time_ns: the sub-second time of the trigger, in ns.
        :param trace_length_nbins: desired length of trace to be loaded from \
        TBB HDF5 files. Does not affect trace size read-in for RFI cleaning
        :type trace_length_nbins: int
        :type LORA_trigger_time_s: int
        :type LORA_trigger_time_ns: int
        """
        self.__trace_length_nbins = trace_length_nbins

        lofar_trace_access = lofar_io.GetLOFARTraces(self.__data_filename,
                                                     self.__metadata_dir,
                                                     LORA_trigger_time_s,
                                                     LORA_trigger_time_ns,
                                                     self.__trace_length_nbins)

        deviating_dipoles_tbb, missing_dipoles_tbb = lofar_trace_access.check_trace_quality()
        logger.info(f"Dipoles with deviating sample number or data length: {deviating_dipoles_tbb}")
        logger.info(f"Dipoles with missing counterparts: {missing_dipoles_tbb}")

        for antenna in self.antennas:
            for dipole in antenna.dipoles:
                raw_trace = lofar_trace_access.get_trace(dipole.get_dipole_id())
                dipole.set_raw_trace(raw_trace)

                if dipole.get_dipole_id() in missing_dipoles_tbb:
                    dipole.set_flagged("Dipole is missing counterpart")
                    antenna.set_flagged("Antenna is missing >= 1 dipole")
                    logger.debug(
                        f"Antenna {antenna.get_antenna_id()} is missing 1 dipole, flagging"
                    )
                elif dipole.get_dipole_id() in deviating_dipoles_tbb:
                    dipole.set_flagged("Dipole deviates in TBB sample number or TBB data length")
                    antenna.set_flagged("Antenna has deviating sample number or data length in >= 1 dipole")
                    logger.debug(
                        f"Antenna {antenna.get_antenna_id()} has deviating sample number or data length, flagging"
                    )
                elif np.max(np.abs(raw_trace)) == 0.0:
                    dipole.set_flagged("Trace is all zero")
                    antenna.set_flagged("Trace is all zero in >= 1 dipole")
                    logger.debug(
                        "Trace is all zero in dipole %s, flagging" % dipole.get_dipole_id()
                    )

        lofar_trace_access.close_file()
        return

    def do_rfi_cleaning(self, retrieve_channels_from_database=None):

        """
        Do RFI cleaning. Assumes self.__data_filename has been set.
        """
        # TODO: Caution: hardcoded value here. Improve this code later.
        rfi_cleaning_trace_chunksize = 65536

        if not self.__data_filename:
            logger.error("Set TBB Data File name and load traces before rfi.")
            raise ValueError

        if (self.__trace_length_nbins < rfi_cleaning_trace_chunksize) or \
                (self.__trace_length_nbins % rfi_cleaning_trace_chunksize != 0):
            logger.error(f"trace length has to be greater than {rfi_cleaning_trace_chunksize} \
                and a multiple of it")
            raise ValueError

        packet = find_rfi.FindRFI_LOFAR(self.__data_filename,
                                        self.__metadata_dir,
                                        self.__trace_length_nbins,
                                        rfi_cleaning_trace_chunksize)

        if retrieve_channels_from_database is not None:
            import psycopg2

            # Open PostgreSQL database
            conn = psycopg2.connect(host='astropg.science.ru.nl', user='crdb', password='crdb', dbname='crdb')
            # Get cursor on database
            cur = conn.cursor()

            table_entries_to_read_out = ['e.eventID',
                                         's.stationname',
                                         'sp.crp_dirty_channels'
                                         ]

            sql_start = "SELECT " + ', '.join(table_entries_to_read_out)

            sql = sql_start + f""" FROM events AS e
                LEFT JOIN eventparameters AS ep ON (e.eventID=ep.eventID)
                INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
                INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
                INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
                INNER JOIN stationparameters AS sp ON (s.stationID=sp.stationID)
                WHERE (s.stationname='{self.__station_name}' AND e.status='CR_FOUND' 
                AND e.eventid={retrieve_channels_from_database}) ORDER BY e.eventID;"""

            cur.execute(sql)

            # Get SQL output
            output_list = cur.fetchall()

            # Decode SQL output (it may contain base64-encoded strings)
            decoded_output_list = decode_sql_output(output_list)

            if not np.all(packet[1] == decoded_output_list[0][2]):
                logger.warning("Dirty channels don't match with the database. Using old ones instead..")

        # TODO: understand avg_antenna_power from FindRFI (eg. packet[3])
        # The current implementation is wrong! The trace in FindRFI is multiplied by a half-Hann before taking the power
        self.__dirty_channels = packet[1] if retrieve_channels_from_database is None else decoded_output_list[0][2]
        self.__dirty_channels_blocksize = packet[2] if retrieve_channels_from_database is None else 65536

        self.__avg_powerspectrum = packet[0]

        freq_Hz = np.fft.rfftfreq(self.__dirty_channels_blocksize, d=1e-6 / self.__clock_frequency_MHz)
        channel_width_Hz = (freq_Hz[1] - freq_Hz[0])

        # TODO: reshape antenna related arrays to be (48, 2) ?
        self.__avg_antenna_power = np.copy(packet[3])
        self.__avg_antenna_power[:, self.__dirty_channels] *= 0.0
        self.__avg_antenna_power = 2 * np.sum(self.__avg_antenna_power, axis=1)

        self.__avg_antenna_power = self.__avg_antenna_power / channel_width_Hz * (
                1 / self.__dirty_channels_blocksize ** 2
        )

        logger.info("Checking for bad antennas from outliers in power")

        median_dipole_power = np.median(self.__avg_antenna_power)
        bad_dipoles = np.where(
            np.logical_or(
                0.5 * median_dipole_power > self.__avg_antenna_power,
                self.__avg_antenna_power > 2 * median_dipole_power
            )
        )[0]

        for index in bad_dipoles:
            ant_index = index // 2  # division by 2 converts dipole numbers to antenna index
            dipole_index = index % 2  # remainder of division is the dipole index (ie 0 or 1)
            if not self.antennas[ant_index].is_flagged():
                self.antennas[ant_index].dipoles[dipole_index].set_flagged("Dipole is outlier in cleaned power")
                self.antennas[ant_index].set_flagged("Antenna has outlier in cleaned power in >= 1 dipole")

        logger.info("Producing cleaned traces")

        self.apply_to_all_traces(self.remove_dirty_channels, self.__dirty_channels,
                                 trace_type='raw', save_type='cleaned', vectorise=True)

    @staticmethod
    def remove_dirty_channels(traces, dirty_channels, axis=-1):
        """
        This vectorised function takes in an array of traces, applies an FFT to them and removes all the dirty channels
        from the resulting array.

        :param traces: Array containing the traces along the dimension indicated by `axis`.
        :type traces: np.ndarray
        :param dirty_channels: List of the dirty channel indices.
        :type dirty_channels: 1d, array_like
        :param axis: Dimension along which the traces are contained in the `traces` array, defaults to -1.
        :type axis: int
        """
        fft_trace = np.fft.rfft(np.swapaxes(traces, axis, -1))  # make the traces are on the last dimension

        fft_trace[..., dirty_channels] *= 0.0  # index dirty channels over last dimension

        return np.swapaxes(np.fft.irfft(fft_trace), axis, -1)  # return with same shape

    def trim_traces(self, new_tracelength_nbins):
        """
        Trims all traces. Time_array, Raw, Clean, Calibrated, Ex,Ey, and Ez.

        Trim traces for all stations to new value, in order to save memory space.
        To be called after RFI cleaning.

        :param new_tracelength_nbins: target trace length value.
        :type new_tracelength_nbins: int
        """
        if new_tracelength_nbins >= self.__trace_length_nbins:
            logger.error(f"new_tracelength_nbins:{new_tracelength_nbins}\
             cannot be >= orig value:{self.__trace_length_nbins}.")
            raise ValueError

        # update tracelength value:
        self.__trace_length_nbins = new_tracelength_nbins

        # trim
        for ant in self.antennas:
            ant.trim_traces(new_tracelength_nbins)

        return

    def __pack_antenna_traces(self, trace_type, not_flagged=True):
        """
        Retrieve all the traces for all the antennas present in the stations, for all polarizations. The function packs
        them in a Numpy array and returns the array. Use the accompanying function
        :py:meth:`.Station.__unpack_antenna_traces` to redistribute the traces over the antennas
        after manipulation.

        **Note**: this function assumes that all antennas have the same dipole polarizations.

        :param trace_type: The type of trace to load from the Dipole instances, must be either "raw", "cleaned"
            or "calibrated".
        :type trace_type: str
        :param not_flagged: If this is set to `True` (the default), only the traces of the non-flagged antennas are
            packed. Make sure to use this consistent with the `not_flagged` parameter
            in :py:meth:`.Station.__unpack_antenna_traces`
        :type not_flagged: bool
        :return: The packed antenna traces, in a Numpy array shaped as (polarizations, antennas, samples).
        """
        polarizations = self.antennas[0].get_polarizations()
        if not_flagged:
            antennas = self.get_good_antennas()
        else:
            antennas = self.antennas

        traces = np.zeros(
            (
                len(polarizations),
                len(antennas),
                len(antennas[0].get_dipole_for_polarization(0).get_raw_trace()),
            )
        )

        for idx, antenna in enumerate(antennas):
            for nr, pol in enumerate(polarizations):
                if trace_type == "raw":
                    traces[nr, idx, :] = antenna.get_dipole_for_polarization(
                        pol
                    ).get_raw_trace()
                elif trace_type == "cleaned":
                    traces[nr, idx, :] = antenna.get_dipole_for_polarization(
                        pol
                    ).get_cleaned_trace()
                elif trace_type == "calibrated":
                    traces[nr, idx, :] = antenna.get_dipole_for_polarization(
                        pol
                    ).get_calibrated_trace()
                else:
                    raise ValueError("Trace type not recognized")

        return traces

    def __unpack_antenna_traces(self, traces, trace_type, not_flagged=True):
        """
        Distributes all the traces in the provided array over the antennas present in the station, using the same
        conventions as :py:meth:`.Station.__pack_antenna_traces`.

        **Note**: this function assumes that the order of the antennas has not been switch after packing, not in the
        traces array and not in the Station object.

        :param traces: All the antenna traces, shaped as (polarizations, antennas, samples)
        :type traces: np.ndarray
        :param trace_type: The type of trace to store the traces in. Must be "raw", "cleaned" or "calibrated".
        :type trace_type: str
        :param not_flagged: If this is set to `True` (the default), the `traces` array is considered to only contain
            traces of antennas which are not flagged. Make sure to use this consistent with the `not_flagged` parameter
            in :py:meth:`.Station.__pack_antenna_traces`.
        :type not_flagged: bool
        :return: None
        """
        polarizations = self.antennas[0].get_polarizations()
        if not_flagged:
            antennas = self.get_good_antennas()
        else:
            antennas = self.antennas

        nr_pol, nr_antenna, _ = traces.shape

        assert len(polarizations) == nr_pol, "Number of polarizations does not match"
        assert len(antennas) == nr_antenna, "Number of antennas does not match"

        for idx in range(nr_antenna):
            for nr in range(nr_pol):
                if trace_type == "raw":
                    antennas[idx].get_dipole_for_polarization(
                        polarizations[nr]
                    ).set_raw_trace(traces[nr, idx, :])
                elif trace_type == "cleaned":
                    antennas[idx].get_dipole_for_polarization(
                        polarizations[nr]
                    ).set_cleaned_trace(traces[nr, idx, :])
                elif trace_type == "calibrated":
                    antennas[idx].get_dipole_for_polarization(
                        polarizations[nr]
                    ).set_calibrated_trace(traces[nr, idx, :])
                else:
                    raise ValueError("Trace type not recognized")

    def __pack_traces_xyz(self, trace_type, not_flagged=True):
        """
        Retrieve the physical traces for all the antennas present in the stations, for the x, y and z polarizations
        (assuming the LOFAR coordinate system). The function packs them in a Numpy array and returns the array. Use the
        accompanying function :py:meth:`.Station.__unpack_traces_xyz` to redistribute the traces
        over the antennas after manipulation.

        :param trace_type: The type of trace to store the traces in. Must be "electric field".
        :type trace_type: str
        :param not_flagged: If this is set to `True` (the default), the `traces` array is considered to only contain
            traces of antennas which are not flagged. Make sure to use this consistent with the `not_flagged` parameter
            in :py:meth:`.Station.__pack_antenna_traces`
        :type not_flagged: bool
        :return: The packed physical traces, in a Numpy array shaped as (xyz, antennas, samples).
        """

        if not_flagged:
            antennas = self.get_good_antennas()
        else:
            antennas = self.antennas

        traces = np.zeros(
            (
                3,  # we have the x, y and z 'polarisations'
                len(antennas),
                len(antennas[0].get_dipole_for_polarization(0).get_raw_trace()),
            )
        )

        for idx, antenna in enumerate(antennas):
            if trace_type == "electric field":
                traces[:, idx, :] = antenna.get_E_trace()  # the current E trace implementation is an array with xyz
            else:
                raise ValueError("Trace type not recognized")

    def __unpack_traces_xyz(self, traces, trace_type, not_flagged=True):
        """
        Distributes all the electric field traces in the provided array over the antennas present in the station, using
        the same conventions as :py:meth:`.Station.__pack_traces_xyz`.

        **Note**: this function assumes that the order of the antennas has not been switch after packing, not in the
        traces array and not in the Station object.

        :param traces: All the antenna traces, shaped as (polarizations, antennas, samples)
        :type traces: np.ndarray
        :param trace_type: The type of trace to store the traces in. Must be "electric field".
        :type trace_type: str
        :param not_flagged: If this is set to `True` (the default), the `traces` array is considered to only contain
            traces of antennas which are not flagged. Make sure to use this consistent with the `not_flagged` parameter
            in :py:meth:`.Station.__pack_traces_xyz`.
        :type not_flagged: bool
        :return: None
        """

        if not_flagged:
            antennas = self.get_good_antennas()
        else:
            antennas = self.antennas

        nr_antenna = traces.shape[1]

        assert len(antennas) == nr_antenna, "Number of antennas does not match"
        assert trace_type == "electric field", "Trace type is not electric field"

        for idx in range(nr_antenna):
            self.antennas[idx].set_E_trace(traces[:, idx, :])

    def do_galactic_calibration(self, timestamp, experiment, cal='new'):
        """
        Apply the galactic calibration to all the antenna's, to each dipole polarization separately. This function
        loads the RFI cleaned traces from the Dipole() instances, so it should only be run after RFI removal has been
        applied. Both the absolute calibration using Galactic noise and the relative calibration between antenna's is
        applied.

        The absolute calibration makes use of a measured calibration curve, which encodes

        #. The conversion from ADC to Volts,
        #. As well as the gains and losses in the amplifiers and coax cables.

        The relative calibration makes sure all the antennas are calibrated to the same reference value. On the other
        hand, the absolute calibration correlates this reference value to the Galactic noise in order to make the units
        physically meaningful. More information can be found in the docstring of the calibration function
        :py:func:`kratos.physics.galactic_calibration.calibrate`.

        :param timestamp: The UNIX timestamp corresponding to the observation.
        :type timestamp: int
        :param experiment: Reference to the antenna set parameters to use.
        :type experiment: str
        :param cal: Set this to 'old' to use the old Galactic calibration values,.
        :type cal: str
        :return: None
        """
        # TODO: implement SystemResponse class to use for calibration curves etc
        # TODO: make experiment parameter consistent with TBB antennaSet?

        # Get the experiment parameters such as latitude and longitude
        with open(
                os.path.join(self.__metadata_dir, "experiment_parameters.txt"),
                "r",
        ) as f:
            all_experiment_parameters = f.readlines()

        experiment_parameters = None
        for line in all_experiment_parameters:
            if line.startswith(experiment):
                experiment_parameters = line.split(", ")

        if experiment_parameters is None:
            logger.error("Experiment was not found in parameter list")
            raise ValueError

        # Get absolute calibration curve
        abs_calibration_curve = np.genfromtxt(
            os.path.join(
                self.__metadata_dir, "calibration", f"{experiment}_galactic_30_80.txt"
            ),
        )

        # Get fitted Fourier coefficients for relative calibration for all polarizations
        if cal == 'new':
            rel_calibration_file = np.genfromtxt(
                os.path.join(
                    self.__metadata_dir, "calibration", f"{experiment}_Fourier_coefficients.txt",
                ),
                dtype=str,
                delimiter=', '
            )
        elif cal == 'old':
            rel_calibration_file = np.genfromtxt(
                os.path.join(
                    self.__metadata_dir, "calibration", f"{experiment}_Fourier_coefficients_old.txt",
                ),
                dtype=str,
                delimiter=', '
            )
        else:
            logger.error("Cal type not recognized")
            raise ValueError
        # rel_calibration_file[0] contains the polarization names

        # Get the traces and polarizations of the antennas
        polarizations = self.antennas[0].get_polarizations()
        traces = self.__pack_antenna_traces(
            trace_type="cleaned"
        )  # shape (polarizations, good antennas, samples)

        # Calibrate the traces per polarization
        calibrated_traces = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            pol = polarizations[i]
            pol_coefficient_index = np.where(rel_calibration_file[0] == f"polarization {pol}")[0]

            rel_calibration_coefficients = rel_calibration_file[1:].astype('f8')[:, pol_coefficient_index]

            calibrated_traces[i, :, :] = physics.galactic_calibration.calibrate(
                timestamp,
                traces[i, :, :],
                self.__clock_frequency_MHz,
                float(experiment_parameters[4]),
                float(experiment_parameters[5]),
                abs_calibration_curve,
                rel_calibration_coefficients,
                debug_power=self.__avg_antenna_power[i::2][self.get_good_antenna_indices()]
            )
        # Distribute calibrated traces back to the antennas
        self.__unpack_antenna_traces(calibrated_traces, "calibrated")

    def do_antenna_unfolding(self, event):

        # Get the  calibrated traces of the antennas
        traces = self.__pack_antenna_traces(
            trace_type="calibrated"
        )
        pol0 = traces[0, :, :]
        pol1 = traces[1, :, :]

        good_ant_indices = self.get_good_antenna_indices()
        positions_ = self.__positions[::2][good_ant_indices]

        # Get the lora direction
        lora_az = event.event_dict["LORA"]["azimuth_rad"]
        lora_zen = event.event_dict["LORA"]["zenith_rad"]
        lora_direction = np.array([lora_az, lora_zen])

        lora_direction *= 180.0 / np.pi  # change to degrees!!

        logger.debug('Going to do pulsefinding with beamforming (unfolding antenna response)')
        packet = unfold_antenna_response.pulsefinding_w_beamforming(pol0,
                                                                    pol1,
                                                                    positions_,
                                                                    lora_direction)

        cr_found = packet[0]
        dominant_pol = packet[1]
        noise_window_start, noise_window_end = packet[2:4]
        signal_window_start, signal_window_end = packet[4:6]

        if not cr_found:
            logger.warning("Cosmic ray was not found in this station")
            return

        nr_antennas_check, good_dipoles = \
            unfold_antenna_response.good_antennas(
                pol0, pol1, noise_window_start, noise_window_end, signal_window_start,
                signal_window_end
            )

        if not nr_antennas_check:
            logger.warning("Minimum number of good antennas was not reached in this station")
            return

        frequencies = self.get_fftfreq_MHz()
        frequencies_HZ = frequencies * 1e6

        new_pulse_direction, dominant_pol, peak_pos = \
            unfold_antenna_response.direction_fit(
                pol0, pol1, positions_, lora_direction, frequencies_HZ, dominant_pol
            )

        packet2 = unfold_antenna_response.plane_wave_fitting(
            pol0, pol1, positions_, new_pulse_direction, peak_pos, dominant_pol
        )

        onsky_0 = packet2[0]
        onsky_1 = packet2[1]
        pulse_direction = packet2[3]

        E_field_traces = unfold_antenna_response.unfold_e_field_xyz(onsky_0, onsky_1, pulse_direction)

        self.__unpack_traces_xyz(E_field_traces, "electric field")
        # TODO: I need to check if this way of writing the traces is indeed ok.

    def get_data_dict_from_members(self):
        """
        Returns a dictionary with keys and values that have to be saved to disk

        :return: dictionary with station keys and values
        :rtype: dict
        """

        data_dict = {"station_name": self.__station_name,
                     "data_filename": self.__data_filename,
                     "antennaset": self.__antenna_set,
                     "antenna_model": self.__antenna_model,
                     "clock_frequency_MHz": self.__clock_frequency_MHz,
                     "dirty_channels": self.__dirty_channels,
                     "dirty_channels_blocksize": self.__dirty_channels_blocksize,
                     "raw_avg_powerspectrum": self.__avg_powerspectrum,
                     "cleaned_avg_powerspectrum": self.get_cleaned_avg_powerspectrum(),
                     "trace_length_nbins": self.__trace_length_nbins,
                     "time_axis_ns": self.__time_axis_ns
                     }

        for i, ant in enumerate(self.antennas):
            data_dict[f"Antenna_{i}"] = ant.get_data_dict_from_members()

        return data_dict

    def set_members_from_data_dict(self, data_dict):
        """
        Accepts a dictionary as input and sets values for members of the class.

        :param data_dict: nested dictionary with station keys and antenna keys
        :type data_dict: dict
        """

        self.__data_filename = data_dict["data_filename"]
        self.__station_name = data_dict["station_name"]
        self.__antenna_set = data_dict["antennaset"]
        self.__antenna_model = data_dict["antenna_model"]
        self.__clock_frequency_MHz = data_dict["clock_frequency_MHz"]
        self.__dirty_channels = data_dict["dirty_channels"]
        self.__dirty_channels_blocksize = data_dict["dirty_channels_blocksize"]
        self.__avg_powerspectrum = data_dict["raw_avg_powerspectrum"]
        self.__trace_length_nbins = data_dict["trace_length_nbins"]
        self.__time_axis_ns = data_dict["time_axis_ns"]

        self.antennas = []

        antenna_keys = [i for i in list(data_dict.keys()) if 'Antenna_' in i]

        for key in antenna_keys:
            new_ant = Antenna(data_dict=data_dict[key])
            self.antennas.append(new_ant)

        return

    def get_raw_avg_powerspectrum(self):
        """
        returns raw, average power spectrum over all dipoles.
        """
        return self.__avg_powerspectrum

    def get_cleaned_avg_powerspectrum(self):
        """
        returns average power spectrum after removing dirty channels.
        """
        spec = np.copy(self.__avg_powerspectrum)
        # ascertain that spectrum was calculated. then assume that dirty channels are calculated as well.
        if len(spec) != 0:
            spec[self.__dirty_channels] *= 0.0
        return spec

    def get_calibration_delay_weights(self):
        """
        Calculate the complex weights associated to the cable delays extracted from the metadata.
        """
        frequencies = np.fft.rfftfreq(self.__trace_length_nbins, d=1e-6 / self.__clock_frequency_MHz)

        phases = calc_interferometer_phase(self.__calibration_delays[:, :, np.newaxis], frequencies)

        weights = phase_to_complex(phases)

        return weights

    def apply_calibration_delays(self, trace_type='raw', save_type=None):
        """
        Apply the cable delays extracted from the metadata to the non-flagged antenna's in frequency space. The delays
        are converted to complex weights, using their phase information.
        """
        weights = self.get_calibration_delay_weights()  # (2, ant, freq)

        all_traces = self.__pack_antenna_traces(trace_type)  # (2, good antenna, samples)
        good_ant_indices = self.get_good_antenna_indices()

        fft_traces = np.fft.rfft(all_traces, axis=-1)
        fft_traces *= weights[:, good_ant_indices]

        applied_traces = np.fft.irfft(fft_traces, axis=-1)

        # If required, store the handled traces back into variables
        if save_type is not None:
            self.__unpack_antenna_traces(applied_traces, save_type)

        return applied_traces
