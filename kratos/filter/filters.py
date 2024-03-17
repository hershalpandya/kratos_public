import numpy as np

from kratos.physics.signal_processing import simple_bandpass, half_hann_window


class Filter:
    """
    Base class for Filter objects. These can easily be applied to all traces in an `Event` or `Station` through the
    corresponding functionalities for these classes.
    """
    def __init__(self, filter_type='filter'):
        self.type = filter_type

    def apply_to_trace(self, trace, axis=-1):
        """
        This is the standard function to apply a filter to a particular trace. Each Filter subclass must reimplement
        this function, as this will be called by an Event to apply to the traces. Note that the variable `trace` will
        most often be a Numpy array, containing multiple traces.

        As such, it should be a `vectorised function <https://numpy.org/doc/stable/glossary.html#term-vectorization>`_,
        in order to benefit from the increased speed Numpy offers. If this is not possible, the keyword `axis` can
        also simply be used to identify the dimension along which the traces are contained. The function
        `numpy.apply_along_axis()` could then be used to apply the 1-D function over each trace.
        """
        raise NotImplementedError


class BandpassFilter(Filter):
    """
    Create a simple bandpass filter between two frequencies. Sets all negative frequencies to zero, as well as all
    frequencies outside the intended range.

    :param sampling_freq: The sampling frequency of the trace to which the filter will be applied, in Hz. Defaults
    to 200 MHz.
    :type sampling_freq: float
    :param lower_freq: The lowest frequency to keep, in Hz. Defaults to 30 MHz.
    :type lower_freq: float
    :param upper_freq: The highest frequency to keep, in Hz. Defaults to 80 MHz.
    :type upper_freq float:
    :param roll_width:
    :type roll_width: float
    """
    def __init__(self, sampling_freq=200e6, lower_freq=30.0e6, upper_freq=80.0e6, roll_width=2.5e6):
        super().__init__(filter_type='bandpass filter')
        self.sample_spacing_s = 1 / sampling_freq
        self.freq_range_Hz = (lower_freq, upper_freq)
        self.roll_width = roll_width

    @staticmethod
    def apply_bandpass(trace, bandpass=None, axis=-1):
        """
        Convenience function to Fourier Transform the trace, multiply it with the bandpass and
        returning the Inverse Fourier Transform of the result.
        """
        trace_swapped = np.swapaxes(trace, axis, -1)  # put the dimension over which to filter last

        fft_temp = np.fft.rfft(trace_swapped)  # computes fft over last dimension by default
        filtered = fft_temp * bandpass  # multiplication is also over last dimension
        filtered_swapped = np.fft.irfft(filtered)

        return np.swapaxes(filtered_swapped, axis, -1)  # return with same shape as input

    def apply_to_trace(self, trace, axis=-1):
        """
        Apply the filter to an array of traces, where the traces are contained along the dimension indicated by `axis`.
        """
        frequencies = np.fft.rfftfreq(trace.shape[axis], d=self.sample_spacing_s)
        my_filter = simple_bandpass(frequencies, *self.freq_range_Hz, roll_width=self.roll_width)

        return self.apply_bandpass(trace, bandpass=my_filter, axis=axis)


class HalfHannFilter(Filter):
    """
    Create a half-Hann window, using the function from
    `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html>`_.
    Note that this is different from a Hamming window.

    This window tapers the beginning and end of a trace, with a raised cosine that touches zero at both ends. A trace
    with length L, will be multiplied with the window `w(n)` defined as

    .. math::
        w(n) = 0.5 - 0.5 \\cos( \\frac{2 \\pi n}{M - 1} ) \\quad &\\text{for} 0 \\leq n < int(L \\cdot p) \\
        w(n) = 1 \\quad  &\\text{for} int(L \\cdot p) \\leq n < int(L \\cdot (1 - p)) \\
        w(n) = 0.5 - 0.5 \\cos( \\frac{2 \\pi n}{M - 1} ) \\quad  &\\text{for} n \\geq int(L \\cdot (1 - p))

    where p is the `half_percent` parameter.

    :param half_percent: Percentage of the trace to taper using the half-Hann window.
    :type half_percent: float
    """
    def __init__(self, half_percent=0.1):
        super().__init__(filter_type='half-Hann window')
        self.half_percent = half_percent

    def apply_to_trace(self, trace, axis=-1):
        """
        Apply the filter to an array of traces, where the traces are contained along the dimension indicated by `axis`.
        """
        trace_swapped = np.swapaxes(trace, axis, -1)  # put the dimension over which to filter last

        my_filter = half_hann_window(trace.shape[axis], half_percent=self.half_percent)

        return np.swapaxes(trace_swapped * my_filter, axis, -1)
