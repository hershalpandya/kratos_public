#!/usr/bin/env python

"""
Author: Brian Hare

A set of tools for modelling the antenna response and for calibrating the amplitude vs frequency of the antennas and
calibrating the absolute amplitude scale for air showers measured at LOFAR. This module is based on `pyCRtools
(Schellart et al) <https://www.astro.ru.nl/software/pycrtools/index.html>`_ and `Detecting cosmic rays with the LOFAR
radio telescope (Nelles et al) <https://arxiv.org/abs/1311.1399>`_.

Note: LBA_ant_calibrator still needs some work.
"""

# internal
import os
from kratos.data_io import storage_paths

# external
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# import metadata as md

pathfinder = storage_paths.PathFinder()
MetaData_directory = os.path.join(
    pathfinder.get_metadata_dir(), "antenna_response_model/"
)


# TODO: improve this so that it has a workspace for N frequencies, so that it doesn't have to allocate memory every call
# TODO: make it so that subsequent calls use same internal memory if length is same (but output doesn't share memory)
class LBAAntennaModel:
    """
    A class encapsulating the antenna model for the Low Band Antennas.
    """

    def __init__(self):
        voltage_theta = np.loadtxt(
            MetaData_directory + "LBA_Vout_theta.txt", skiprows=1
        )
        voltage_phi = np.loadtxt(MetaData_directory + "LBA_Vout_phi.txt", skiprows=1)

        voltage_theta_responce = voltage_theta[:, 3] + 1j * voltage_theta[:, 4]
        voltage_phi_responce = voltage_phi[:, 3] + 1j * voltage_phi[:, 4]

        freq_start = 10.0 * 1.0e6
        freq_step = 1.0 * 1.0e6
        num_freq = 101

        theta_start = 0.0
        theta_step = 5.0
        num_theta = 19

        phi_start = 0.0
        phi_step = 10.0
        num_phi = 37

        frequency_samples = np.arange(num_freq) * freq_step + freq_start
        theta_samples = np.arange(num_theta) * theta_step + theta_start
        phi_samples = np.arange(num_phi) * phi_step + phi_start

        voltage_theta_responce = voltage_theta_responce.reshape(
            (num_freq, num_theta, num_phi)
        )
        voltage_phi_responce = voltage_phi_responce.reshape(
            (num_freq, num_theta, num_phi)
        )

        self.theta_responce_interpolant = RegularGridInterpolator(
            (frequency_samples, theta_samples, phi_samples), voltage_theta_responce
        )
        self.phi_responce_interpolant = RegularGridInterpolator(
            (frequency_samples, theta_samples, phi_samples), voltage_phi_responce
        )

    def JonesMatrix(self, frequency, zenith, azimuth):
        """
        Return the Jones Matrix for a single frequency (in Hz), for a wave with a zenith and azimuth angle in degrees.
        Dot the jones matrix with the electric field vector, first component of vector is Zenith component of
        electric field and second component is azimuthal electric field, then the first component of the resulting
        vector will be voltage on odd  (X) antenna and second component will be voltage on even (Y) antenna.
        Returns identity matrix where frequency is outside of 10 to 100 MHz.

        :param frequency: frequency
        :type frequency: np.float64
        :param zenith: zenith angle
        :type zenith: np.float64
        :param azimuth: azimuth angle
        :type azimuth: np.float64
        :return: jones_matrix
        :type: np.ndarray
        """

        jones_matrix = np.zeros((2, 2), dtype=complex)

        if (
            frequency < 10.0e6 or frequency > 100.0e6
        ):  # if frequency is outside of range, then return some invertable nonsense
            jones_matrix[0, 0] = 1.0
            jones_matrix[1, 1] = 1.0
            return jones_matrix

        # calculate for X dipole
        azimuth += 135  # put azimuth in coordinates of the X antenna
        while azimuth > 360:  # normalize the azimuthal angle
            azimuth -= 360
        while azimuth < 0:
            azimuth += 360

        jones_matrix[0, 0] = self.theta_responce_interpolant(
            [frequency, zenith, azimuth]
        )
        jones_matrix[0, 1] = -1 * self.phi_responce_interpolant(
            [frequency, zenith, azimuth]
        )  # I don't really know why this -1 must be here

        # calculate for Y dipole
        azimuth += 90.0  # put azimuth in coordinates of the Y antenna
        while azimuth > 360:  # normalize the azimuthal angle
            azimuth -= 360
        while azimuth < 0:
            azimuth += 360
        jones_matrix[1, 0] = -1 * self.theta_responce_interpolant(
            [frequency, zenith, azimuth]
        )  # I don't really know why this -1 must be here
        jones_matrix[1, 1] = self.phi_responce_interpolant([frequency, zenith, azimuth])

        return jones_matrix

    def JonesMatrix_MultiFreq(self, frequencies, zenith, azimuth, out=None):
        """
        Same as JonesMatrix, except that frequencies is expected to be an array.
        Returns an array of jones matrices.

        :param frequencies: frequencies
        :type frequencies: np.ndarray
        :param zenith: zenith angle
        :type zenith: np.float64
        :param azimuth: azimuth angle
        :type azimuth: np.float64
        :return: out_JM
        :type: np.ndarray
        """

        if out is None:
            out_JM = np.zeros((len(frequencies), 2, 2), dtype=complex)
        else:
            out_JM = out

        good_frequencies = np.logical_and(frequencies > 10.0e6, frequencies < 100e6)
        num_freqs = np.sum(good_frequencies)

        points = np.zeros((num_freqs, 3))  # figure out how to not need this
        points[:, 0] = frequencies[good_frequencies]
        points[:, 1] = zenith

        # calculate for X dipole
        points[:, 2] = azimuth + 135  # put azimuth in coordinates of the X antenna
        while np.any(points[:, 2] > 360):  # normalize the azimuthal angle
            points[:, 2][points[:, 2] > 360] -= 360
        while np.any(points[:, 2] < 0):
            points[:, 2][points[:, 2] < 0] += 360

        out_JM[good_frequencies, 0, 0] = self.theta_responce_interpolant(points)
        out_JM[good_frequencies, 0, 1] = -1 * self.phi_responce_interpolant(points)

        # calculate for Y dipole
        points[:, 2] += 90.0  # put azimuth in coordinates of the Y antenna
        while np.any(points[:, 2] > 360):  # normalize the azimuthal angle
            points[:, 2][points[:, 2] > 360] -= 360
        while np.any(points[:, 2] < 0):
            points[:, 2][points[:, 2] < 0] += 360

        out_JM[good_frequencies, 1, 0] = -1 * self.theta_responce_interpolant(points)
        out_JM[good_frequencies, 1, 1] = self.phi_responce_interpolant(points)

        # set the frequencies outide 10 to 100 MHz to just identity matix
        out_JM[np.logical_not(good_frequencies), 0, 0] = 1.0
        out_JM[np.logical_not(good_frequencies), 1, 1] = 1.0

        #        fi = np.argmin( np.abs(frequencies-60.0E6) )

        return out_JM


def unravelAntennaResponce(self, zenith, azimuth):
    antenna_model = LBAAntennaModel().JonesMatrix  # (self, frequency, zenith, azimuth):
    """
    Given a direction to source (azimuth off X and zenith from Z, in degrees ), if call this function, then apply_GalaxyCal MUST also be applied to the data
    Note that this function assumes the data is LBA_outer, which has flipped polarizations compared to LBA inner.
    """

    # jones_matrices = self.antenna_model(self.frequencies, zenith, azimuth)

    # inverse_jones_matrix = invert_2X2_matrix_list( jones_matrices )

    # apply the Jones matrix.  Note that the polarities (even and odd) are flipped)
    # zenith_component = self.odd_pol_FFT*inverse_jones_matrix[:,0,0] +  self.even_pol_FFT*inverse_jones_matrix[:,0,1]
    # azimuth_component = self.odd_pol_FFT*inverse_jones_matrix[:,1,0] +  self.even_pol_FFT*inverse_jones_matrix[:,1,1]

    # self.even_pol_FFT = zenith_component
    # self.odd_pol_FFT = azimuth_component


def invert_2X2_matrix_list(matrices):
    """
    If matrices is an array of 2x2 matrices, then return the array of inverse matrices.

    :param matrices: matrices
    :type matrices: np.ndarray
    :return: out
    :type: np.ndarray
    """
    num = len(matrices)
    out = np.zeros((num, 2, 2), dtype=matrices.dtype)

    out[:, 0, 0] = matrices[:, 1, 1]
    out[:, 0, 1] = -matrices[:, 0, 1]
    out[:, 1, 0] = -matrices[:, 1, 0]
    out[:, 1, 1] = matrices[:, 0, 0]

    determinants = (
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )

    out /= determinants[:, np.newaxis, np.newaxis]

    return out


def return_unfolded_traces(even_pol_FFT, odd_pol_FFT, pulse_direction, frequencies):
    """
    Returns the zenith and azimuth components of unfolded traces given pulse direction
    and frequencies.

    :param even_pol_FFT: polarization 0 fft
    :type even_pol_FFT: np.ndarray
    :param odd_pol_FFT: polarization 1 fft
    :type odd_pol_FFT: np.ndarray
    :param pulse_direction: pulse direction
    :type pulse_direction: np.ndarray
    :param frequencies: frequencies
    :type frequencies: np.ndarray
    :return: zenith_component
    :type: np.ndarray
    :return: azimuth_component
    :type: np.ndarray
    """
    am = LBAAntennaModel().JonesMatrix_MultiFreq

    jones_matrices = am(
        np.abs(frequencies), 90 - pulse_direction[1], 90 - pulse_direction[0]
    )  # zenith, azimuth (phi north from east)
    # inverse_jones_matrix = antenna_model.invert_2X2_matrix_list( np.asarray([jones_matrices]) )
    inverse_jones_matrix = invert_2X2_matrix_list(jones_matrices)

    zenith_component = np.zeros([len(odd_pol_FFT), len(odd_pol_FFT[0])], dtype=complex)
    azimuth_component = np.zeros([len(odd_pol_FFT), len(odd_pol_FFT[0])], dtype=complex)

    for i in np.arange(len(odd_pol_FFT)):
        zenith_component[i] = (
            odd_pol_FFT[i] * inverse_jones_matrix[:, 0, 0]
            + even_pol_FFT[i] * inverse_jones_matrix[:, 0, 1]
        )
        azimuth_component[i] = (
            odd_pol_FFT[i] * inverse_jones_matrix[:, 1, 0]
            + even_pol_FFT[i] * inverse_jones_matrix[:, 1, 1]
        )

    return zenith_component, azimuth_component


def unfold_model(timeseries_0, timeseries_1, direction):
    """
    Main interface to unfold the antenna model. Takes as input
    timeseries for both dipoles and pulse direction and returns
    the zenith and azimuth components in time domain.

    :param timeseries_0: polarization 0 fft
    :type timeseries_0: np.ndarray
    :param timeseries_1: polarization 1 fft
    :type timeseries_1: np.ndarray
    :param direction: pulse direction
    :type direction: np.ndarray
    :return: onsky_0
    :type: np.ndarray
    :return: onsky_1
    :type: np.ndarray
    """
    # array of instrumental 0/1 dipole traces
    even_pol_FFT = np.fft.rfft(timeseries_0)
    odd_pol_FFT = np.fft.rfft(timeseries_1)
    frequencies = np.fft.rfftfreq(len(timeseries_0[0]), d=5e-9)

    fft_onsky_0, fft_onsky_1 = return_unfolded_traces(
        even_pol_FFT, odd_pol_FFT, direction, frequencies
    )
    onsky_0 = np.fft.irfft(fft_onsky_0)
    onsky_1 = np.fft.irfft(fft_onsky_1)

    return onsky_0, onsky_1
