import numpy as np
from scipy.optimize import fmin_powell

from kratos.physics.coordinate_transformations import spherical_to_cartesian

lightspeed = 299792458.0


def GeometricDelayFarField(position, direction, length):
    """Returns the time delay of the far field given position,
    direction and track length.

    :param position: position
    :type position: np.ndarray
    :param direction: direction
    :type direction: np.ndarray
    :param length: length
    :type length: float
    :return: delay
    :type: float
    """
    delay = (direction[0] * position[0] + direction[1] * position[1] + direction[2] * position[2]) / length / lightspeed
    return delay


def minibeamformer(fft_data, frequencies, positions, direction):
    """Returns beamformed signal given fft, frequencies,
    positions and direction.

    :param fft_data: data from fast fourier transform
    :type fft_data: np.ndarray
    :param frequencies: frequencies
    :type frequencies: np.ndarray
    :param positions: position
    :type positions: np.ndarray
    :param direction: direction
    :type direction: np.ndarray
    :return: output
    :type: np.ndarray
    """
    # adapted from hBeamformBlock
    nantennas = len(positions)
    nfreq = len(frequencies)
    output = np.zeros([len(frequencies)], dtype=complex)

    norm = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])

    for a in np.arange(nantennas):
        delay = GeometricDelayFarField(positions[a], direction, norm)
        # print(delay)

        for j in np.arange(nfreq):
            real = 1.0 * np.cos(2 * np.pi * frequencies[j] * delay)
            imag = 1.0 * np.sin(2 * np.pi * frequencies[j] * delay)
            de = complex(real, imag)
            output[j] = output[j] + fft_data[a][j] * de
            # *it_out += (*it_fft) * polar(1.0, (2*np.pi)*((*it_freq) * delay));
    return output


def geometric_delays(antpos, sky):
    """Returns geometric delays in a matrix.

    :param antpos: antenna positions
    :type antpos: np.ndarray
    :param sky: position in the sky
    :type sky: np.ndarray
    :return: delays
    :type: np.ndarray
    """
    distance = np.sqrt(sky[0] ** 2 + sky[1] ** 2 + sky[2] ** 2)
    delays = (np.sqrt(
        (sky[0] - antpos[0]) ** 2 + (sky[1] - antpos[1]) ** 2 + (sky[2] - antpos[2]) ** 2) - distance) / lightspeed
    return delays


def beamformer(fft_data, frequencies, delay):
    """Returns beamformed spectrum given fft data, frequencies and time delays.

    :param fft_data: data from fast fourier transform
    :type fft_data: np.ndarray
    :param frequencies: frequencies
    :type frequencies: np.ndarray
    :param delay: time delays
    :type delay: np.ndarray
    :return: output
    :type: np.ndarray
    """
    nantennas = len(delay)
    nfreq = len(frequencies)
    output = np.zeros([len(frequencies)], dtype=complex)

    for a in np.arange(nantennas):
        for j in np.arange(nfreq):
            real = 1.0 * np.cos(2 * np.pi * frequencies[j] * delay[a])
            imag = 1.0 * np.sin(2 * np.pi * frequencies[j] * delay[a])
            de = complex(real, imag)
            output[j] = output[j] + fft_data[a][j] * de
    return output


def directionFitBF(fft_data, frequencies, antpos, start_direction, maxiter):
    """Returns direction of beamformed signal in cartesian coordinates and corresponding timeseries.

    :param fft_data: data from fast fourier transform
    :type fft_data: np.ndarray
    :param frequencies: frequencies
    :type frequencies: np.ndarray
    :param antpos: antenna positions
    :type antpos: np.ndarray
    :param start_direction: starting direction for fitting
    :type start_direction: np.ndarray
    :param maxiter: maximum number of iteration for fitting procedure
    :type maxiter: int
    :return: fit_direction
    :type: np.ndarray
    :return: timeseries
    :type: np.ndarray
    """
    def negative_beamed_signal(direction):
        rho = 1.0
        theta = 90 - direction[1]
        phi = 360 - direction[0]
        direction_cartesian = spherical_to_cartesian(rho, theta, phi)
        # print("antpos = ", antpos[1][1])
        # print("direction_cartesian = ", direction_cartesian)
        delays = geometric_delays(antpos, direction_cartesian)
        # print("delays = ", delays)
        out = beamformer(fft_data, frequencies, delays)
        timeseries = np.fft.irfft(out)
        return -100 * np.max(timeseries ** 2)

    fit_direction = fmin_powell(negative_beamed_signal, np.asarray(start_direction), maxiter=maxiter, xtol=1.0)

    rho = 1.0
    theta = (2 * np.pi) - np.radians(fit_direction[1])
    phi = (2 * np.pi) - np.radians(fit_direction[0])
    direction_cartesian = spherical_to_cartesian(rho, theta, phi)
    delays = geometric_delays(antpos, direction_cartesian)
    out = beamformer(fft_data, frequencies, delays)
    timeseries = np.fft.irfft(out)

    return fit_direction, timeseries


def return_minibeamformed_data(timeseries_data, positions, direction):
    """Returns beamformed timeseries for each polarization given dipole timeseries,
    dipole positions and pulse direction.

    :param timeseries_data: timeseries
    :type timeseries_data: np.ndarray
    :param positions: antenna positions
    :type positions: np.ndarray
    :param direction: pulse direction
    :type direction: np.ndarray
    :return: beamformed_timeseries
    :type: np.ndarray
    """
    fft_data = np.fft.rfft(timeseries_data)
    frequencies = np.fft.rfftfreq(len(timeseries_data[0]), d=5e-9)

    positions_ = positions[::2]
    x, y, z = spherical_to_cartesian(1, (np.pi / 2) - np.radians(direction[1]),
                                         (np.pi / 2) - np.radians(direction[0]))
    direction_cartesian = np.array([x, y, z])

    beamed_fft = minibeamformer(fft_data, frequencies, positions_, direction_cartesian)

    beamformed_timeseries = np.fft.irfft(beamed_fft)

    return beamformed_timeseries