import numpy as np
from scipy.signal import hilbert
from scipy.signal import resample
from kratos.physics.beamformer import return_minibeamformed_data
from kratos.physics.beamformer import directionFitBF
from kratos.physics.planewave import directionForHorizontalArray
from kratos.physics.planewave import timeDelaysFromDirection
from kratos.physics.coordinate_transformations import space_angle
from kratos.physics import antenna_model
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(1)

def pulsefinding_w_beamforming(pol0, pol1, antenna_positions, pulse_direction):
    """
    Performs pulsefinding with beamforming on calibrated traces. It uses an initial estimate of
    the signal pulse direction and returns a boolean whether cosmic ray is found or not along
    with the corresponding signal, noise windows and dominant dipole polarization.

    :param pol0: polarization 0 traces
    :type pol0: np.ndarray
    :param pol1: polarization 1 traces
    :type pol1: np.ndarray
    :param antenna_positions: antenna positions
    :type antenna_positions: np.ndarray
    :param pulse_direction: pulse direction
    :type pulse_direction: np.ndarray
    :return: cr_found
    :type: boolean
    :return: dominant_pol
    :type: int
    :return: noise_window_start
    :type: int
    :return: noise_window_end
    :type: int
    :return: signal_window_start
    :type: int
    :return: signal_window_end
    :type: int
    """

    beamformed_timeseries_0 = return_minibeamformed_data(
        pol0, antenna_positions, pulse_direction
    )
    beamformed_timeseries_1 = return_minibeamformed_data(
        pol1, antenna_positions, pulse_direction
    )
    ## Look for significant pulse in beamformed signal
    mid = 2 ** 16 / 2  # trace length - do not hardcode
    # search_window_start = int(mid - 1000)  # i made up the window size
    # search_window_end = int(mid + 3000)

    noise_window_start = int(0)
    noise_window_end = int(mid - 5000)
    good_signal = 3  # pick acceptable snr
    window = 500

    analytic_signal_0 = hilbert(beamformed_timeseries_0)
    analytic_signal_1 = hilbert(beamformed_timeseries_1)
    amplitude_envelope_0 = np.abs(analytic_signal_0)
    amplitude_envelope_1 = np.abs(analytic_signal_1)

    # find dominant polarization
    peak0 = np.max(amplitude_envelope_0)
    peak1 = np.max(amplitude_envelope_1)

    if peak0 > peak1:
        dominant_pol = 0
    else:
        dominant_pol = 1

    if dominant_pol == 0:
        signal_window_start = (
                np.argmax(amplitude_envelope_0) - window / 2
        )  # I made up the window size
        signal_window_end = int(np.argmax(amplitude_envelope_0) + window / 2)
    else:
        signal_window_start = int(
            np.argmax(amplitude_envelope_1) - window / 2
        )  # I made up the window size
        signal_window_end = int(np.argmax(amplitude_envelope_1) + window / 2)

    # print("signal window start = ", signal_window_start)
    # print("signal window end = ", signal_window_end)

    rms_0 = np.sqrt(
        np.mean((amplitude_envelope_0[noise_window_start:noise_window_end]) ** 2)
    )
    rms_1 = np.sqrt(
        np.mean((amplitude_envelope_1[noise_window_start:noise_window_end]) ** 2)
    )

    peak_0 = np.max(amplitude_envelope_0[signal_window_start:signal_window_end])
    peak_1 = np.max(amplitude_envelope_1[signal_window_start:signal_window_end])

    snr_0 = peak_0 / rms_0
    snr_1 = peak_1 / rms_1

    # max_sample_0=np.argmax(amplitude_envelope_0[signal_window_start:signal_window_end])
    # max_sample_1=np.argmax(amplitude_envelope_1[signal_window_start:signal_window_end])
    if snr_0 >= good_signal or snr_1 >= good_signal:
        cr_found = True
    else:
        cr_found = False
    # if no CR found, exit station loop
    return cr_found, dominant_pol, noise_window_start, noise_window_end, signal_window_start, signal_window_end


def good_antennas(pol0, pol1, start_window_noise, end_window_noise, start_window_signal, end_window_signal):
    """
    Starts with the hilbert envelopes for each dipole polarization and finds antennas with acceptable snr.
    Returns a boolean whether the number of dipoles is enough for the station to be considered further or not,
    and the number of dipoles with sufficient signal in the station.

    :param pol0: polarization 0 traces
    :type pol0: np.ndarray
    :param pol1: polarization 1 traces
    :type pol1: np.ndarray
    :param start_window_noise: starting index of the noise window
    :type start_window_noise: int
    :param end_window_noise: last index of the noise window
    :type end_window_noise: int
    :param start_window_signal: starting index of the signal window
    :type start_window_signal: int
    :param end_window_signal: last index of the signal window
    :type end_window_signal: int
    :return: nr_antennas_check
    :type: boolean
    :return: good_dipoles
    :type: int
    """

    single_analytic_signal_0 = hilbert(pol0)
    single_analytic_signal_1 = hilbert(pol1)

    single_amplitude_envelope_0 = np.abs(single_analytic_signal_0)
    single_amplitude_envelope_1 = np.abs(single_analytic_signal_1)

    # rms_all = np.zeros([2, len(pol0)])
    # peak_all = np.zeros([2, len(pol0)])
    # snr_all = np.zeros([2, len(pol0)])
    good_dipoles = 0
    good_signal = 3  # pick acceptable snr
    min_number_good_antennas = 3

    for i in np.arange(len(pol0)):
        rms0 = np.sqrt(
            np.mean(
                (single_amplitude_envelope_0[i][start_window_noise:end_window_noise]) ** 2
            )
        )
        rms1 = np.sqrt(
            np.mean(
                (single_amplitude_envelope_1[i][start_window_noise:end_window_noise]) ** 2
            )
        )
        peak0 = np.max(
            single_amplitude_envelope_0[i][start_window_signal:end_window_signal]
        )
        peak1 = np.max(
            single_amplitude_envelope_1[i][start_window_signal:end_window_signal]
        )
        snr0 = peak0 / rms0
        snr1 = peak1 / rms1

        if snr0 >= good_signal or snr1 >= good_signal:
            good_dipoles = good_dipoles + 1

    if good_dipoles >= min_number_good_antennas:
        nr_antennas_check = True
    else:
        nr_antennas_check = False

    # if fewer than min_number_good_antennas found, exit station loop
    return nr_antennas_check, good_dipoles


def direction_fit(pol0, pol1, antenna_positions, pulse_direction, frequencies, dominant_pol):
    """
    Perfoms the direction fitting loop. Uses as input both polarization traces, dipole positions,
    a starting estimate of the pulse direction, corresponding frequencies and the dominant polarization
    as calculated from beamforming. Returns updated values for pulse direction and dominant
    polarization along with the index of the pulse peak.

    :param pol0: polarization 0 traces
    :type pol0: np.ndarray
    :param pol1: polarization 1 traces
    :type pol1: np.ndarray
    :param antenna_positions: antenna positions that coincide with dipole positions
    :type antenna_positions: np.ndarray
    :param pulse_direction: pulse direction
    :type pulse_direction: np.ndarray
    :param frequencies: frequencies
    :type frequencies: np.ndarray
    :param dominant_pol: index of the dominant polarization
    :type dominant_pol: int
    :return: new_pulse_direction
    :type: np.ndarray
    :return: dominant_pol
    :type: int
    :return: peak_pos
    :type: np.ndarray
    """

    onsky_0, onsky_1 = antenna_model.unfold_model(pol0, pol1, pulse_direction)
    positions_0 = antenna_positions
    positions_1 = antenna_positions

    new_pulse_direction = pulse_direction  # pulse_direction
    old_pulse_direction = (0.0, 0.0)
    maxiter = 30

    direction_difference = np.asarray([100, 100])
    fft_onsky_0 = np.fft.rfft(onsky_0)
    fft_onsky_1 = np.fft.rfft(onsky_1)

    logger.debug('Pulse direction is set to %s before starting beamformed fit' % new_pulse_direction)

    while direction_difference[0] > 1 and direction_difference[1] > 1:
        old_pulse_direction = new_pulse_direction

        # not really necessary to do the fit in both polarizations?
        direction_fit_1, timeseries1 = directionFitBF(
            fft_onsky_1, frequencies, positions_1, new_pulse_direction, maxiter
        )
        direction_fit_0, timeseries0 = directionFitBF(
            fft_onsky_0, frequencies, positions_0, new_pulse_direction, maxiter
        )
        if dominant_pol == 0:
            new_pulse_direction = direction_fit_0
        else:
            new_pulse_direction = direction_fit_1

        direction_difference = np.abs(
            np.asarray(
                [
                    old_pulse_direction[0] - new_pulse_direction[0],
                    old_pulse_direction[1] - new_pulse_direction[1],
                ]
            )
        )
        logger.debug('Difference after another fit iteration is %s;' % direction_difference)
        logger.debug('Direction after this fit iteration is %s;' % new_pulse_direction)
        
    # find dominant polarization
    analytic_signal_0 = hilbert(pol0)
    analytic_signal_1 = hilbert(pol1)

    amplitude_envelope_0 = np.abs(analytic_signal_0)
    amplitude_envelope_1 = np.abs(analytic_signal_1)

    peak0 = np.max(amplitude_envelope_0)
    peak1 = np.max(amplitude_envelope_1)

    if peak0 > peak1:
        dominant_pol = 0
        peak_pos = np.argmax(amplitude_envelope_0)
    else:
        dominant_pol = 1
        peak_pos = np.argmax(amplitude_envelope_1)

    return new_pulse_direction, dominant_pol, peak_pos


def plane_wave_fitting(pol0, pol1, antenna_positions, pulse_direction, peak_pos, dominant_pol):
    """
    Perfoms the plane wave fitting procedure. Uses as input both polarization traces, dipole positions,
    the pulse direction from direction fitting and the corresponding pulse peak and dominant polarization.
    Returns voltage traces for both polarizations, time delays, an updated pulse direction and the
    angular difference between the input pulse direction and the resulting one.

    :param pol0: polarization 0 traces
    :type pol0: np.ndarray
    :param pol1: polarization 1 traces
    :type pol1: np.ndarray
    :param antenna_positions: antenna positions that coincide with dipole positions
    :type antenna_positions: np.ndarray
    :param pulse_direction: pulse direction
    :type pulse_direction: np.ndarray
    :param peak_pos: pulse peak
    :type peak_pos: np.ndarray
    :param dominant_pol: index of the dominant polarization
    :type dominant_pol: int
    :return: on_sky_0
    :type: np.ndarray
    :return: on_sky_1
    :type: np.ndarray
    :return: expected_delays
    :type: np.ndarray
    :return: pulse_direction
    :type: np.ndarray
    :return: angular_diff_deg
    :type: float
    """

    first_direction = pulse_direction
    onsky_0, onsky_1 = antenna_model.unfold_model(pol0, pol1, first_direction)
    positions_0 = antenna_positions
    positions_1 = antenna_positions

    resample_factor = 16

    window_start = int(int(peak_pos - 2000))  # I made up the window size
    window_end = int(int(peak_pos + 2000))  # I made up the window size
    print("window_start = ", window_start)
    print("window_end = ", window_end)

    signal_window_start = int(int(peak_pos - 500) - window_start)  # I made up the window size
    signal_window_end = int(int(peak_pos + 500) - window_start)  # I made up the window size
    print("signal_window_start = ", signal_window_start)
    print("signal_window_end = ", signal_window_end)

    noise_window_start = int(0)
    noise_window_end = 1500
    print("noise_window_start = ", noise_window_start)
    print("noise_window_end = ", noise_window_end)

    ndipoles = int(pol0.shape[0])
    delay = np.zeros([ndipoles])
    rms0 = np.zeros([ndipoles])
    rms1 = np.zeros([ndipoles])
    peak0 = np.zeros([ndipoles])
    peak1 = np.zeros([ndipoles])
    snr0 = np.zeros([ndipoles])
    snr1 = np.zeros([ndipoles])
    ### caution here---------------------------------
    for p in np.arange(ndipoles):
        # don't need to look at the whole trace
        signal_0 = onsky_0[p][window_start:window_end]
        signal_1 = onsky_1[p][window_start:window_end]

        f0 = resample(signal_0, len(signal_0) * resample_factor)
        f1 = resample(signal_1, len(signal_1) * resample_factor)

        # Apply Hilbert transform
        analytic_signal_0 = hilbert(f0)
        analytic_signal_1 = hilbert(f1)
        # get envelope
        amplitude_envelope_0 = np.abs(analytic_signal_0)
        amplitude_envelope_1 = np.abs(analytic_signal_1)

        # Find signal to noise ratio, maximum, position of maximum and rms
        rms0[p] = np.sqrt(np.mean(
            (amplitude_envelope_0[resample_factor * noise_window_start: resample_factor * noise_window_end]) ** 2))
        rms1[p] = np.sqrt(np.mean(
            (amplitude_envelope_1[resample_factor * noise_window_start: resample_factor * noise_window_end]) ** 2))
        peak0[p] = np.max(
            amplitude_envelope_0[resample_factor * signal_window_start: resample_factor * signal_window_end])
        peak1[p] = np.max(
            amplitude_envelope_1[resample_factor * signal_window_start: resample_factor * signal_window_end])
        snr0[p] = peak0[p] / rms0[p]
        snr1[p] = peak1[p] / rms1[p]

        # Find time delay from start of block
        position_of_max0 = np.argmax(
            amplitude_envelope_0[resample_factor * signal_window_start: resample_factor * signal_window_end])
        position_of_max1 = np.argmax(
            amplitude_envelope_1[resample_factor * signal_window_start: resample_factor * signal_window_end])

        if dominant_pol == 0:
            position_of_max = position_of_max0
        else:
            position_of_max = position_of_max1
        # find in terms of original sampling
        position_of_max = (position_of_max / resample_factor + signal_window_start) + window_start
        delay[p] = position_of_max * 5e-9

    if dominant_pol == 0:
        goodtimes = delay[snr0 > 3]
        goodpositions = positions_0[snr0 > 3]
        # goodantennas=allantennas[snr0>3]
    else:
        goodtimes = delay[snr1 > 3]
        goodpositions = positions_1[snr1 > 3]
        # goodantennas=allantennas[snr1>3]

    goodtimes = goodtimes - goodtimes[0]  # normalize to reference antenna
    goodcount = len(goodtimes)
    # check that at least 3 antennas are good
    goodantennas = np.arange(goodcount)

    indicesOfGoodAntennas = goodantennas

    goodtimes_fit = goodtimes
    goodpositions_fit = goodpositions
    # iteratively remove antennas for best fit
    while True:
        print("goodpositions_fit = ", goodpositions_fit)
        print("goodtimes_fit = ", goodtimes_fit)
        goodcount = len(indicesOfGoodAntennas)
        (az, el) = directionForHorizontalArray(goodpositions_fit.ravel(), goodtimes_fit)
        expected_delays = timeDelaysFromDirection(goodpositions_fit.ravel(), (az, el))
        expected_delays -= expected_delays[0]
        residual_delays = goodtimes_fit - expected_delays
        spread = np.std(residual_delays)
        k = 2  # rms factor
        goodSubset = np.where(abs(residual_delays - np.mean(residual_delays)) < k * spread)
        if len(goodSubset[0]) == goodcount:
            break
        else:
            goodcount = len(goodSubset[0])
            tmp = indicesOfGoodAntennas[goodSubset[0]]
            indicesOfGoodAntennas = tmp
            goodpositions_fit = goodpositions[indicesOfGoodAntennas]
            goodtimes_fit = goodtimes[indicesOfGoodAntennas]

    cartesianDirection = [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    meanDirection = cartesianDirection

    # redo and save for all antennas
    expected_delays = timeDelaysFromDirection(goodpositions.ravel(), (az, el))  # need positions as flat 1-D array
    expected_delays -= expected_delays[0]  # subtract ref ant

    # check that fit converges
    maximum_angular_diff = 0.5
    pulse_direction = [az, el]  # relation between elevation
    last_direction = first_direction
    angular_diff_deg = space_angle(
        pulse_direction[::-1], last_direction[::-1]
    )

    return onsky_0, onsky_1, expected_delays, pulse_direction, angular_diff_deg


def unfold_e_field_xyz(onsky_0, onsky_1, pulse_direction):
    """
    Transforms the voltage traces to electric field with x, y and z components.
    Uses as input the voltage traces for both polarizations and the pulse direction
    and returns the x,y and z electric field vector components in one array.

    :param onsky_0: polarization 0 traces
    :type onsky_0: np.ndarray
    :param onsky_1: polarization 1 traces
    :type onsky_1: np.ndarray
    :param pulse_direction: pulse direction
    :type pulse_direction: np.ndarray
    :return: E
    :type: np.ndarray
    """

    phi = pulse_direction[0]
    theta = pulse_direction[1]
    ndipoles = len(onsky_0)
    trace_length = onsky_0.shape[1]

    Ex = np.zeros((ndipoles, trace_length))
    Ey = np.zeros((ndipoles, trace_length))
    Ez = np.zeros((ndipoles, trace_length))

    # loop over all dipoles
    for p in np.arange(ndipoles):
        Ex[p] = np.cos(theta) * np.cos(phi) * onsky_0[p] - np.sin(phi) * onsky_1[p]
        Ey[p] = np.cos(theta) * np.sin(phi) * onsky_0[p] + np.cos(phi) * onsky_1[p]
        Ez[p] = - np.sin(theta) * onsky_0[p]

    E = np.zeros((3, ndipoles, trace_length))
    E[0, :, :] = Ex
    E[1, :, :] = Ey
    E[2, :, :] = Ez

    return E