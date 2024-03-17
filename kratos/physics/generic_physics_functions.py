import numpy as np
from scipy.constants import mu_0, c


def power(time_series, apply_rfft=True, axis=-1):
    """
    Calculate the power contained in a time series, using the RFFT and impedance of free space.

    :param time_series: Time samples
    :type time_series: np.ndarray
    :param apply_rfft: Whether to apply the RFFT (ie are the samples provided in time or frequency space), defaults to
        True.
    :type apply_rfft: bool, optional
    :param axis: The axis over which to compute the power.
    :type axis: int
    :return: Power in the time series.
    """
    # Define impedance of free space
    Z_0 = mu_0 * c

    if apply_rfft:
        time_series = np.fft.rfft(time_series, axis=axis)

    return np.sum(2 * np.abs(time_series) ** 2 / Z_0, axis=axis)


def get_time_ns_array_for_trace(clock_frequency, trace_length_bins=65536):
    """
    Relative time axis, starting at 0 for first sample in traces read-in.

    :param clock_frequency: 1/dt for 1 bin of the trace, in MHz.
    :type clock_frequency: float
    :param trace_length_bins: Number of bins in the trace.
    :type trace_length_bins: int
    :return: time_ns array
    :rtype: numpy.ndarray
    """
    ns_per_sample = 1.0e3 / clock_frequency
    return np.arange(trace_length_bins) * ns_per_sample


def get_fftfreq_MHz(clock_frequency, trace_length_bins=65536):
    """
    Frequency axis for numpy real fft applied on a trace.

    :param clock_frequency: 1/dt for 1 bin of the trace, in MHz.
    :type clock_frequency: float
    :param trace_length_bins: Number of bins in the trace.
    :type trace_length_bins: int
    :return: Array of frequencies, in MHz.
    :rtype: numpy.ndarray
    """
    seconds_per_sample = 1.0e-6 / clock_frequency
    freq = np.fft.rfftfreq(trace_length_bins, seconds_per_sample)
    return freq / 1.0e6  # for MHz freq


def calc_interferometer_phase(freq, time):
    """
    Returns the interferometer phase in radians for a given frequency and time.
    
    This function was copied from
    `PyCRTools <https://gitlab.science.ru.nl/lofar_crksp/pycrtools/-/blob/master/src/PyCRTools/mMath.cc>`_.
    """
    return 2 * np.pi * freq * time


def phase_to_complex(phase):
    """
    Converts a real phase to a complex number, with amplitude of unity.
    
    This function was copied from
    `PyCRTools <https://gitlab.science.ru.nl/lofar_crksp/pycrtools/-/blob/master/src/PyCRTools/mMath.cc>`_.
    """
    return np.exp(1j * phase)
