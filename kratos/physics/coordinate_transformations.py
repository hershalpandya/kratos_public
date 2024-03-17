import numpy as np

from radiotools import helper
from radiotools.coordinatesystems import cstrafo


# Some radiotools reimplementations
def get_magnetic_field_vector(site=None):
    """
    Reimplementation of function from `helper` module in `radiotools`, with custom magnetic field vectors.
    Currently, the implemented vectors are:

    * LOFAR site (values from 2015): keyword 'lofar' (default)
    * Auger site: keyword 'auger'
    * Moores bay: keyword 'mooresbay'
    * Summit station in Greenland: keyword 'summit'
    * South Pole Arianna station: keyword 'southpole'

    Returns the geomagnetic field vector in Gauss, where x points to geographic East and y towards geographic North.
    """
    magnetic_fields = {'auger': np.array([0.00871198, 0.19693423, 0.1413841]),
                       'mooresbay': np.array([0.058457, -0.09042, 0.61439]),
                       'summit': np.array([-.037467, 0.075575, -0.539887]),  # Summit station, Greenland
                       'southpole': np.array([-0.14390398, 0.08590658, 0.52081228]),  # position of SP arianna station
                       'lofar': np.array([0.004675, 0.186270, -0.456412])  # values from 2015
                       }
    if site is None:
        site = 'auger'
    return magnetic_fields[site]


def get_angle_to_magnetic_field_vector(zenith, azimuth, site=None):
    """
    Reimplementation of function from `helper` module in `radiotools`.

    Returns the angle between shower axis, defined by `zenith` and `azimuth` and the magnetic field at `site`. The
    function :py:func: `get_magnetic_field_vector` is used to retrieve the latter.
    """
    magnetic_field = get_magnetic_field_vector(site=site)
    v = helper.spherical_to_cartesian(zenith, azimuth)
    return helper.get_angle(magnetic_field, v)


def spherical_to_cartesian(rho, theta, phi):
    """
    Reimplementation of function from `helper` module in `radiotools`, which adds the `rho` parameter to the function
    call.

    Converts a position from spherical to cartesian coordinates, following the
    `ISO convention <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_.

    :param rho: The radial distance to the origin, often denoted as :math:`{\\rho}`.
    :type rho: float
    :param theta: The polar angle :math:`{\\theta}`, in degrees.
    :type theta: float
    :param phi: The azimuthal angle :math:`{\\phi}`, in degrees.
    :type phi: float
    :return: Position vector in cartesian coordinates, containing x, y, z, in same units as `rho`.
    :type: np.ndarray
    """
    return rho * helper.spherical_to_cartesian(np.deg2rad(theta), np.deg2rad(phi))


class CoordinateTransformer(cstrafo):
    """
    Reimplementation of the `cstrafo` class from the `radiotools` package. The differences are the unit convention for
    the angles and the implementation of custom magnetic field vectors. For the list of implemented vectors, refer to
    the documentation of the function :py:func: `get_magnetic_field_vector` .
    """

    def __init__(self, zenith, azimuth, site=None):
        """
        Initialise transformer instance with air shower direction and site name.

        :param zenith: Zenith angle of the air shower signal, in degrees.
        :type zenith: float
        :param azimuth: Azimuth angle of the air shower signal, in degrees.
        :type azimuth: float
        :param site: The keyword to use for getting the magnetic field vector, defaults to `None`.
        :type site: str
        """
        super().__init__(np.deg2rad(zenith), np.deg2rad(azimuth),
                         magnetic_field_vector=get_magnetic_field_vector(site=site))


# Extra coordinate conversion functions
def space_angle(dir1, dir2):
    """
    Calculates space angle between two directions, defined by pairs of (zenith, azimuth).

    :param dir1: The first direction/pair of angles.
    :type dir1: array_like
    :param dir2: The second direction/pair of angles.
    :type dir2: array_like
    :return: The angle between the (unit) vectors defined by the angles `dir1` and `dir2` in degrees.
    """
    v1 = spherical_to_cartesian(1.0, *dir1)
    v2 = spherical_to_cartesian(1.0, *dir2)
    return np.rad2deg(helper.get_angle(v1, v2))
