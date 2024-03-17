import glob
import logging
import os

import numpy as np

class Event:
    def __init__(self,
                 timetamps=None,
                 event_id=None,
                 ):
        """
        If event_id provided, loads event info from json file.
        If timestamp provided, find and process event from root file.
        
        KM: how to handle new vs old versions of LORA root files?

        :param pathfinder: instance of class PathFinder() with paths set.
        :type pathfinder: PathFinder()
        :param event_id: unique event identifier. default:None.
        :type event_id: int
        """


        #theta=0
        #elevation=0
        #phi=0
        #fit_theta=0
        #fit_elevation=0
        #fit_phi=0
        #fit_theta_err=0
        #fit_phi_err=0
        #x_core=0
        #y_core=0
        #x_core_err=0
        #y_core_err=0
        #z_core=0
        #UTC_min=0
        #nsec_min=0
        #energy=0
        #energy_err=0
        #Rm=0
        #fit_elevation_err=0
        #fit_phi_err=0
        #Ne=0
        #Ne_err=0
        #CorCoef_xy=0
        #Ne_RefA=0
        #NeErr_RefA=0
        #Energy_RefA=0
        #EnergyErr_RefA=0
        #direction_flag=0
        #event_flag=0






    def read_attenuation(self):
        """
        Read attenuation from atm
        KM: for NKG fit. maybe overkill
        """
        return

    def find_counts(self):
        """
        Find total ADC counts within a time window for scintillator traces.  Use station.detector
        """
        return
    
    def retrive_sat_signal(self):
        """
        Recover information for a saturated signal
        """
        return

    def get_signal_arrival_time(self):
        """
        Find arrival time of signal from a trace.  
        KM: Use station.detector. signal calibration happens on the station level 
        """
        return

    def get_event_timestamp(self):
        """
        tbd- needs to be done for V1 and V2
        """
        return

    def cal_event_timestamp(self):
        """
        calibrate timing of arrival times, find time of first hit for event timestamp
        """
        return

    def do_arrival_time_diff(self):
        """
        Find relative times of arrival from all detectors 
        """
        return  

    def do_arrival_direction(self):
        """
        Plane wave calculation for arrival direction: move to physics
        NB. not a fit!
        """
        return
    
    def do_COM_core(self):
        """
        Find core based on center of mass: move to physics
        """
        return

    def find_density(self):
        """
        Find charge deposit in each detector
        """
        return
    
    def func_plane(self):
        """
        Function to be used in plane wave fit.
        """
        return

    def fit_arrival_direction(self):
        """
        Do a minimization to find arrival direction.
        KM: Currently uses root functions
        """
        return

    def theta_phi(self):
        """
        Coordinate transformation from ground plane to shower plane.
        Move to physics
        """
        return

    def back_theta_phi():
        """
        Coordinate transformation from shower plane to ground plane.
        Move to physics
        """
        








