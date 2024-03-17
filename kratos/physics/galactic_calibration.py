import datetime
import numpy as np

from scipy.interpolate import interp1d
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u

from kratos.physics import power


def calibrate(timestamp, data, sampling_freq, latitude, longitude, abs_calibration, rel_calibration, debug_power=None):
    """
    Apply the calibration to the traces of a set of antennas, with all the same polarization. Both relative calibration,
    which depends on the polarization and absolute one, which is applied to each antenna separately.

    Further details are described in this `overview <https://arxiv.org/pdf/1311.1399.pdf>`_, and also this
    `paper <https://arxiv.org/pdf/1903.05988.pdf>`_ .

    :param longitude: Longitude in units of degrees (towards east).
    :type longitude: float
    :param latitude: Latitude in units of degrees (towards north).
    :type latitude: float
    :param sampling_freq: The frequency at which the traces are sampled, in MHz.
    :type sampling_freq: float
    :param timestamp: UTC timestamp of the event.
    :type timestamp: inTypeError: Text reading control character must be a single unicode character or None;t
    :param data: Array containing antenna traces, shaped as (antennas, samples).
    :type data: np.ndarray
    :param abs_calibration: Calibration curve to use for absolute calibration.
    :type abs_calibration: np.ndarray
    :param rel_calibration: Fourier coefficients to use for relative calibration.
    :type rel_calibration: np.ndarray
    :param debug_power: Array with the power in each antenna, currently used for debugging.
    :type debug_power: np.ndarray
    :return: Calibrated traces, shaped as (antennas, samples).
    """
    # Find the sidereal time for the LOFAR site
    dt_object = datetime.datetime.utcfromtimestamp(timestamp)
    observing_location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
    observing_time = Time(dt_object, scale="utc", location=observing_location)
    local_time = observing_time.sidereal_time("apparent").hour

    # Apply calibration to each polarization separately
    FFT_freq_Hz = np.fft.rfftfreq(data.shape[1], d=1e-6 / sampling_freq)
    FFT_data = np.apply_along_axis(np.fft.rfft, 1, data, norm='forward')  # "forward" normalises FFT with sample number
    FFT_data_cal = np.zeros_like(FFT_data)

    FFT_data_cal[:, :] = (
            FFT_data[:, :]
            * absolute_calibration(FFT_freq_Hz, FFT_data, abs_calibration)
            * relative_calibration(FFT_freq_Hz, FFT_data, rel_calibration, local_time, trace_power=debug_power)
    )

    data_calibrated = np.fft.irfft(FFT_data_cal, norm='forward')

    return data_calibrated


def relative_calibration(fft_freq, fft_traces, calibration_coefficients, local_sidereal_time, trace_power=None):
    """
    Calculate the relative calibration for a set of antennas, given the Fourier coefficients for the curve of the
    galactic noise power they observe as a function of the local sidereal time. The relative calibration makes sure all
    the given antennas are calibrated to the same reference power. It is not frequency dependent, unlike the absolute
    calibration.

    :param fft_freq: The frequencies samples in `fft_traces`, in Hz.
    :type fft_freq: np.ndarray
    :param local_sidereal_time: The local sidereal time of the observation, in hours.
    :type local_sidereal_time: float
    :param fft_traces: Array containing all the FT antenna traces of single polarization, shaped as
        (antenna, frequency samples).
    :type fft_traces: np.ndarray
    :param calibration_coefficients: The Fourier coefficients of the LST dependent Galactic power curve.
    :type calibration_coefficients: np.ndarray
    :param trace_power: An array containing the power in each trace of `fft_traces`. If not set, this will be
        calculated using the traces.
    :type trace_power: np.ndarray
    :return: The scaling factor per antenna, in an array with the same shape as `fft_traces`.
    """
    # Calculate the power in the signals
    channel_width_Hz = (fft_freq[1] - fft_freq[0])
    if trace_power is None:
        trace_power = np.apply_along_axis(power, 1, fft_traces, apply_rfft=False)
        trace_power /= channel_width_Hz

    # Calculate Galactic power noise
    # -> the local sidereal time runs from 0 to 24 (it is calculated from the Earth angle), so normalise it to 2 * pi
    galactic_noise_power = fourier_series(local_sidereal_time / 24.0 * 2 * np.pi, calibration_coefficients)

    print(galactic_noise_power * channel_width_Hz)

    # Calculate the correction factor per antenna
    scale = galactic_noise_power / trace_power
    scale[np.where(scale == np.inf)] = 0.0  # RFI cleaned frequencies have 0 power, resulting in `inf` in scale

    return np.ones_like(fft_traces) * np.sqrt(scale[:, np.newaxis])


def absolute_calibration(fft_freq, fft_traces, calibration_curve):
    """
    Calculate the absolute calibration for a single dipole trace, using the provided calibration curve. The curve should
    be a 1-D array, containing the calibration values for frequencies from 0 MHz up to `len(calibration_curve)` MHz, in
    steps of 1 MHz.

    :param fft_freq: The frequencies samples in `fft_traces`, in Hz.
    :type fft_freq: np.ndarray
    :param fft_traces: Array containing all the FT antenna traces of single polarization, shaped as
        (antenna, frequency samples).
    :type fft_traces: np.ndarray
    :param calibration_curve: Calibration curve to use, frequency samples are assumed to be in intervals of 1 MHz.
    :type calibration_curve: np.ndarray
    :return: The correction factor per frequency, in an array with the same shape as `fft_traces`.
    """
    # Set the calibration curve frequencies
    calibration_frequencies_Hz = np.arange(len(calibration_curve)) * 1e6

    # Interpolate the curve between the frequency positions
    f = interp1d(calibration_frequencies_Hz, calibration_curve)

    # Apply the interpolation to the sampled frequencies and return
    return np.ones_like(fft_traces) * f(fft_freq)[np.newaxis, :]


def fourier_series(x, p):
    """
    Evaluates a partial Fourier series:

    .. math:: F(x) \\approx \\frac{a_{0}}{2} + \\sum_{n=1}^{\\mathrm{order}} a_{n} \\sin(nx) + b_{n} \\cos(nx)
    """
    r = p[0] / 2
    order = int((len(p) - 1) / 2)
    for i in range(order):
        n = i + 1
        r += p[2 * i + 1] * np.sin(n * x) + p[2 * i + 2] * np.cos(n * x)
    return r


def read_line(file, line):
    """
    Convenience function to read a specific line from a file.

    :param file: Path to file to read from.
    :type file: Path-like str
    :param line: The line to return.
    :type line: int
    :return: The line with number `line` in the file.
    """
    with open(file, "r") as f:
        content = f.readlines()
    return content[line]
