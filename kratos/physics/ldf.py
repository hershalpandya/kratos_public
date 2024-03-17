import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.special import gamma
from coordinate_transformations import get_angle_to_magnetic_field_vector, CoordinateTransformer
# from leastsqbound import *


# Defining functions
def simple_gauss(p, x, y):
    return p[3]*np.exp(- ((x-p[0])**2+(y-p[1]) ** 2) / p[2] ** 2)


def simple_error_function(p, x, y, data):
    return simple_gauss(p, x, y) - data


def LDF(p, x, y):
    # Warning: This function is defined a little ugly with fixed parameters.
    return p[3]*np.exp(-((x-p[0])**2+(y-p[1])**2)/p[2]**2)-p[5]*np.exp(- ((x-(p[0]+p[4]))**2+(y-p[1])**2)/ (np.exp(2.788+0.008*p[2]))**2)

def error_function(p, x, y, data, sigma):
    return (LDF(p, x, y) - data) / sigma



def ldf(eventid=None,
        antenna_positions=None,
        signal_power=None,
        noise_power=None,
        pulse_direction=None,
        particle_core=np.array([0., 0., 0.]),
        particle_direction=np.array([0., 0.]),
        particle_densities=np.ones(20),
        signal_windowsize=11,
        flag_outliers=True,
        flag_outliers_sigma=5,
        flag_outliers_remaining=40,
        additional_uncertainty=0.,
        scale_signals=1,
        n_iterations_uncertainties=300,
        core_correction=(28.58, -7.88),
        energy_conversion=(25.45, 0.477),
        xmax_conversion=(230., 0.91, 0.008),
        full_atmosphere=1036.0,
        lora_lambda=220,
        lora_energy_constants=(1.23, 0.95),
        parameter_restrictions=(0.1, 0.8),
        debug=False,
        plot_type="png",
        save_plots=False,
        plot_prefix="",
        plot_publication=False,
        plotlist=[],
    ):
    """ Fit two-dimensional LDF as described in A. Nelles et al.
    Astroparticle Physics 60, p.13-24 (2014), A parameterization for
    the radio emission of air showers as predicted by CoREAS simulations
    and applied to LOFAR measurements A. Nelles et al. The radio emission
    pattern of air showers as measured with LOFAR - a tool for the
    reconstruction of the energy and the shower maximum , A. Nelles et al.,
    submitted to JCAP.

    :param eventid: event ID
    :type eventid: int
    :param antenna_positions: antenna positions  on the ground
    :type antenna_positions: array [Nantennasx3]
    :param signal_power: measured signals in power. Units are
                         fixed for integrated power in J.
    :type signal_power: array [Nantennasx3]
    :param: noise_power: measured noise in power
    :type noise_power: array [Nantennasx3]
    :param pulse_direction: measured arrival direction of air shower pulse
                            (zenith and azimuthal)
    :type pulse_direction: array [2]
    :param particle_core: Core estimate from LORA on ground (x,y,z)
    :type particle_core: numpy array[2]
    :param particle_direction: measured arrival direction of air shower pulse
                            (zenith and azimuthal)
    :type particle_direction: array[2]
    :param particle_densities: measured particle densities of 20 LORA detectors
    :type particle_densities: numpy array[20]
    :param signal_windowsize: bins over which the signal is integrated
    :type signal_windowsize: 11
    :param flag_outliers: do a second iteration of the fit and remove outliers
    :type flag_outliers: bool
    :param flag_outliers_sigma: remove outliers with this sigma
    :type flag_outliers_remaining: float
    :param additional_uncertainty: relative fraction of additional uncertainty
    :type additional_uncertainty: float
    :param scale_signals: Scales signals to handy values. Should
                          not be used, if working with calibrated data.
    :type scale_signals: float
    :param n_iterations_uncertainties: number of iterations for estimating
                                       the parameter uncertainty
    :type n_iterations_uncertainties: int
    :param core_correction: convert C parameter to core position with these
                            factors (offset, sinusiodial factor)"
    :type core_correction: array[2]
    :param xmax_conversion: converting a parameter to energy with these factors
    :type xmax_conversion: array[2]
    :param full_atmosphere: height of the atmosphere above LOFAR in g/cm^2
    :type full_atmosphere: float
    :param lora_lambda: converting sigma parameter to Xmax with these factors
    :type lora_lambda: array[3]
    :param lora_energy_constants: Constants for LORA [a,b] to obtain energy
                                  from particles (old but robust method)
    :type lora_energy_constants: array[2]
    :param parameter_restrictions: boundaries for fit if first iteration
                                   diverges
    :type parameter_restrictions: array[2]
    :param debug: produce debug output
    :type debug: bool
    :param plot_type: define plot type
    :type plot_type: string
    :param save_plots: store plots
    :type save_plots: bool
    :param plot_prefix: prefix for plots
    :type plot_prefix: string
    :param plot_publication: create publication ready figures, removes titles
                             and collections, returns individual figures
    :type plot_publication: bool
    :param plotlist: list of plots
    :type plotlist: list
    :return ldf_fit_output: output dictionary for fit parameters
    :type ldf_fit_output: dictionary
    :return ldf_fit_core: store new core position separately
    :type ldf_fit_core: numpy array[3]
    :return ldf_fit_energy: store an energy estimate from the radio
                            reconstruction
    :type ldf_fit_energy: float
    :return ldf_fit_energy_particle: store an energy estimate from the
                                     new shower geometry and the particle data
    :type ldf_fit_energy_particle: float
    :return ldf_fit_xmax: store an xmax estimate from the radio reconstruction
    :type ldf_fit_xmax: float
    :return ldf_fit_quality: store an estimator for the quality
    :type ldf_fit_quality: bool
    """

    # ---------------------------------
    # Preparatory steps
    # --------------------------------

    # Fix unphysical negative values
    signal_power[np.where(signal_power < 0)] = 0.

    # Calculate the uncertainty from the intergrated power
    # (depends on windowsize)

    uncertainty_x = np.sqrt(4*noise_power[:, 0]*signal_power[:, 0]
                      + 2*signal_windowsize*noise_power[:, 0]**2)
    uncertainty_y = np.sqrt(4*noise_power[:, 1]*signal_power[:, 1]
                      + 2*signal_windowsize*noise_power[:, 1]**2)
    uncertainty_z = np.sqrt(4*noise_power[:, 2]*signal_power[:, 2]
                      + 2*signal_windowsize*noise_power[:, 2]**2)

    total_power = signal_power[:, 0] + signal_power[:, 1] + signal_power[:, 2]
    uncertainty_noise = uncertainty_x + uncertainty_y + uncertainty_z

    # Scale to reasonable numbers
    total_power *= scale_signals
    uncertainty_noise *= scale_signals

    # Fudging: Add appropriate uncertainties from calibration (probably 2%, default now 0.)
    uncertainty_noise += additional_uncertainty * np.mean(total_power)

    # Converting into cartesian coordinates and radians
    zenith = np.radians(pulse_direction[0])
    azimuth = np.radians(pulse_direction[1])

    # ---------------------------------
    # Changing coordinate system
    # --------------------------------

    # Taking LORA core as first guess and shift coordinate system
    if debug:
        print("LORA initial value", particle_core)
        print("If further outside than 500m, it will be \
              shifted to the center of the superterp")

    if np.abs(particle_core[0]) > 500:
        particle_core[0] = 0.
    if np.abs(particle_core[1]) > 500:
        particle_core[1] = 0.

    pos_uvw = get_pos_shower_plane(antenna_positions, particle_core[0], particle_core[1], zenith, azimuth)

    # --------------------------------
    # Prefit iteration
    # --------------------------------

    # Prefit for initial guess values
    simple_initguess = [0., 0., 160., total_power.max()]
    params_prefit = (pos_uvw[:, 0], pos_uvw[:, 1], total_power)

    # Prefit
    p1, success_1 = opt.leastsq(simple_error_function,
                                simple_initguess,
                                args=params_prefit)

    if debug:
        print("---> LDF Prefit parameters")
        print("Core shift", p1[0], p1[1])
        print("Sigma", p1[2])
        print("Scaling", p1[3])

    # ---------------------------------
    # Main fit iteration
    # --------------------------------

    # check for resonable pre fit parameters to avoid NaNs
    if p1[3] < 0:
        p1[3] = 1.

    # Inital values from pre fit
    params = (pos_uvw[:, 0], pos_uvw[:, 1], total_power, uncertainty_noise)
    initguess = [p1[0], p1[1], abs(p1[2]), p1[3], -50., 0.29*p1[3]**0.994]

    # Brute forcing sigma region to avoid getting stuck in local minimum
    initguess[2] = 150
    p, cov_x, infodict, msg, ier = opt.leastsq(error_function, initguess,
                                               args=params, full_output=True)
    nfev = infodict['nfev']
    chi_test_3 = np.sum((total_power - LDF(p, pos_uvw[:, 0], pos_uvw[:, 1])) ** 2
                        / uncertainty_noise ** 2)

    initguess[2] = 200
    p, cov_x, infodict, msg, ier = opt.leastsq(error_function, initguess,
                                               args=params, full_output=True)
    nfev = infodict['nfev']
    chi_test_2 = np.sum((total_power - LDF(p, pos_uvw[:, 0], pos_uvw[:, 1])) ** 2
                        / uncertainty_noise ** 2)

    # If initial fit already diverged, setting initial parameter smaller
    if p1[2] > 300:
        p1[2] = 300

    initguess = [p1[0], p1[1], p1[2], p1[3], -50., 0.29*p1[3]**0.994]
    p, cov_x, infodict, msg, ier = opt.leastsq(error_function, initguess,
                                               args=params, full_output=True)
    nfev = infodict['nfev']
    chi_test_1 = np.sum((total_power - LDF(p, pos_uvw[:, 0], pos_uvw[:, 1])) ** 2
                        / uncertainty_noise ** 2)

    if debug:
        print("Resulting chi2 for different initial values [150, 200, 300 or true initial value]",
              chi_test_1, chi_test_2, chi_test_3)
        print("If other values improve on initial guess, use those")

    if np.round(chi_test_1, 1) > np.round(chi_test_3, 1):
        p1[2] = 150
        initguess = [p1[0], p1[1], p1[2], p1[3], -50., 0.29*p1[3]**0.994]
        p, cov_x, infodict, msg, ier = opt.leastsq(error_function,
                                                   initguess,
                                                   args=params,
                                                   full_output=True)
        nfev = infodict['nfev']

    if (np.round(chi_test_1, 1) > np.round(chi_test_2, 1)):
        p1[2] = 200
        initguess = [p1[0], p1[1], p1[2], p1[3], -50., 0.29*p1[3]**0.994]
        p, cov_x, infodict, msg, ier = opt.leastsq(error_function,
                                                   initguess,
                                                   args=params,
                                                   full_output=True )
        nfev = infodict['nfev']

    if debug:
        print("---> Mainfit parameters")
        print("Core shift", p[0], p[1])
        print("Sigma", p[2])
        print("Scaling", p[3])
        print("Offset", p[4])
        print("2. scaling", np.round(p[5]/p[3], 3))

    # --------------------------------------------------
    # Entering in restricted fitting, if fit diverged
    # ---------------------------------------------------

    redo_fit = False
    if np.round(p[5]/p[3], 3) < parameter_restrictions[0]:
        redo_fit = True
    if np.round(p[5]/p[3], 3) > parameter_restrictions[1]:
        redo_fit = True
    if p[4] > 0:
        redo_fit = True
    if p[4] < -200:
        redo_fit = True

    if redo_fit:

        low_a = 0.15*p[3]
        high_a = 0.75*p[3]

        bounds = [(None, None), (None, None), (None, None), 
                  (None, None), (-140, -10.), (low_a, high_a)]

        try:
            p, cov_x, infodict, msg, ier = leastsqbound(error_function,
                                                        initguess,
                                                        args=params,
                                                        bounds=bounds,
                                                        full_output=1)
        except:
            p, sucess = leastsqbound(error_function,
                                     initguess,
                                     args=params,
                                     bounds=bounds,
                                     full_output=0)
            print("Warning: bound fit did not work with covariance")

    if debug:
        print("---> Fitting with fixed parameters")
        print("Core shift", p[0], p[1])
        print("Sigma", p[2])
        print("Scaling", p[3])
        print("Offset", p[4])
        print("2. scaling", np.round(p[5]/p[3], 3))

    # -----------------------------------
    # Second fit iteration with flagging
    # ------------------------------------

    f_residuals = np.abs(LDF(p, pos_uvw[:, 0], pos_uvw[:, 1]) - total_power) / uncertainty_noise

    # Flagging according to input variable
    to_flag = np.where(f_residuals > flag_outliers_sigma)
    not_flag = np.where(f_residuals <= flag_outliers_sigma)

    if not_flag[0].shape[0] < flag_outliers_remaining:
        flag_outliers = False
        if debug:
            print("Too many outliers, no second iteration with flagging performed")

    if flag_outliers:

        flagged_pos_uvw = np.copy(pos_uvw)
        flagged_pos_uvw = flagged_pos_uvw[to_flag]
        if debug:
            print("Number of flagged antennas", flagged_pos_uvw.shape)
        flagged_total_p = total_power[to_flag]

        pos_uvw = pos_uvw[not_flag]
        total_power = total_power[not_flag]
        uncertainty_noise = uncertainty_noise[not_flag]

        # Fitting using initialguess from previous fit

        params = (pos_uvw[:, 0], pos_uvw[:, 1], total_power, uncertainty_noise)
        p, cov_x, infodict, msg, ier = opt.leastsq(error_function, initguess, args=params, full_output=1)
        nfev = infodict['nfev']

        redo_fit = False
        if np.round(p[5]/p[3], 3) < 0.1:
            redo_fit = True
        if np.round(p[5]/p[3], 3) > 0.8:
            redo_fit = True
        if p[4] > 0:
            redo_fit = True
        if p[4] < -200:
            redo_fit = True

        if redo_fit:
            low_a = 0.15*p[3]
            high_a = 0.75*p[3]

            bounds = [(None, None), (None, None), (None, None), (None, None), (-140, -10.), (low_a, high_a)]

            try:
                p, cov_x, infodict, msg, ier = leastsqbound(error_function, initguess, args=params, bounds=bounds, full_output=1)
            except:
                p, sucesss = leastsqbound(error_function, initguess, args=params, bounds=bounds, full_output=0)
                print("Warning: bound fit did not work with covariance")

    else:
        flagged_pos_uvw = []
        flagged_total_p = []

    # Calculate chi2 again
    chi2 = np.sum((total_power - LDF(p, pos_uvw[:, 0], pos_uvw[:, 1])) ** 2 / uncertainty_noise ** 2)
    ndof = (len(total_power) - 4)
    red_chi2 = chi2/ndof

    if debug:
        print("Chi2 of final fit",  red_chi2)

    # -------------------------------------------
    # Uncertianty estimation for the parameters
    # -------------------------------------------

    p0_hist = np.zeros(n_iterations_uncertainties)
    p1_hist = np.zeros(n_iterations_uncertainties)
    p2_hist = np.zeros(n_iterations_uncertainties)
    p3_hist = np.zeros(n_iterations_uncertainties)
    p4_hist = np.zeros(n_iterations_uncertainties)
    p5_hist = np.zeros(n_iterations_uncertainties)
    chi_hist = np.zeros(n_iterations_uncertainties)

    for i in range(n_iterations_uncertainties):

        rand = np.random.randn(total_power.shape[0])
        total_p_rand = total_power + rand * uncertainty_noise
        params = (pos_uvw[:, 0], pos_uvw[:, 1], total_p_rand, uncertainty_noise)
        if redo_fit:
            p_rand, success = leastsqbound(error_function, initguess, args=params, bounds=bounds)
        else:
            p_rand, success = opt.leastsq(error_function, initguess, args=params)
        chi_2_rand = np.sum((total_p_rand-LDF(p_rand, pos_uvw[:, 0], pos_uvw[:, 1])) ** 2 / uncertainty_noise ** 2)

        p0_hist[i] = p_rand[0]
        p1_hist[i] = p_rand[1]
        p2_hist[i] = p_rand[2]
        p3_hist[i] = p_rand[3]
        p4_hist[i] = p_rand[4]
        p5_hist[i] = p_rand[5]
        chi_hist[i] = chi_2_rand/len(total_p_rand)

    if debug:
        print("CoreX", np.mean(p0_hist), '+/-', np.std(p0_hist), "m")
        print("CoreY",  np.mean(p1_hist), '+/-', np.std(p1_hist), "m")
        print("Sigma+", np.mean(p2_hist), '+/-', np.std(p2_hist), "m")
        print("Scaling + ", np.std(p3_hist), np.std(p3_hist)/np.mean(p3_hist)*100.)
        print("Offset", np.mean(p4_hist), '+/-', np.std(p4_hist), "m")
        print("Scaling + ", np.std(p5_hist), np.std(p5_hist)/np.mean(p5_hist)*100.)

        ratio = (p5_hist/p3_hist)
        print(np.mean(ratio))

        # Additional histograms can be filled, if debugging of parameter uncertainties is needed

#        plt.figure()
#        plt.hist(p0_hist,bins=20)
#        plt.xlabel('Core X')
#        plt.figure()
#        plt.hist(p1_hist,bins=20)
#        plt.xlabel('Core Y')
#        plt.figure()
#        plt.hist(p2_hist,bins=20)
#        plt.xlabel('Sigma +')
#        plt.figure()
#        plt.hist(p3_hist,bins=20)
#        plt.xlabel('Scaling + ')
#        plt.figure()
#        plt.hist(p4_hist,bins=40,range=(-120,0))
#        plt.xlabel('offset')
#        plt.figure()
#        plt.hist(p5_hist,bins=20)
#        plt.xlabel('Scaling -')
#        plt.figure()
#        plt.hist(ratio,bins=10)
#        plt.xlabel('Scaling Ratio')
#        plt.figure()
#        plt.hist(chi_hist,bins=20)
#        plt.xlabel('Chi Square')

    # -------------------------------------------------
    # Output Paramaters
    out_sigma = p[2]
    out_energy = p[3]
    out_core_x = p[0] - (core_correction[0]+core_correction[1]*np.sin(azimuth))
    out_core_y = p[1]
    out_core_x_un = p[0]
    out_core_y_un = p[1]
    # Retransforming new core to ground plane and original coordinates

    new_core = get_ground([out_core_x, out_core_y, 0.], particle_core[0], particle_core[1], zenith, azimuth)

    new_core_x  = new_core[0]
    new_core_y = new_core[1]

    # ------------------------------------------------
    # Refitting LORA data for new core position

    lora_fit_function = lambda p, x: 
                           p[0]/2/np.pi/p[1]/p[1] * np.power(x / p[1], p[2] - 2) * np.power(1 + x / p[1], p[2] - 4.5) * gamma(4.5 - p[2]) / gamma(p[2]) / gamma(4.5 - 2 * p[2])
    lora_error_function = lambda p, data, err, lpos, cx, cy, az, zen: 
                           (lora_ldf(p[0], p[1], 1.7, lpos, cx, cy, az, zen) - data) / err

    def lora_ldf(nch, rm, s, lpos, cx, cy, az, zen):
        pos_lora_uvw = get_pos_shower_plane(lpos, cx, cy, zen, az)
        r = np.sqrt(pos_lora_uvw[:, 0]*pos_lora_uvw[:, 0]+pos_lora_uvw[:, 1]* pos_lora_uvw[:, 1])
        return lora_fit_function([nch, rm, s], r)

    def get_distance(lpos, cx, cy, az, zen):
        pos_lora_uvw = get_pos_shower_plane(lpos, cx, cy, zen, az)
        r = np.sqrt(pos_lora_uvw[:, 0] * pos_lora_uvw[:, 0] + pos_lora_uvw[:, 1] * pos_lora_uvw[:, 1])
        return r

    def fit_lora_ldf(core_x, core_y, azimuth, zenith, particles,
                     lora_x=(11.21, -29.79, -57.79, -3.79, -120.79, -82.79, -162.79, -134.79, 78.21, 155.21, 112.21, 133.21, 74.21, 118.21, 41.21, 80.21, -53.79, 3.21, -2.79, -48.79),
                     lora_y=(-94.07, -83.07, -125.07, -158.07, 3.93, -40.07, -26.07, -74.07, 75.93, 50.93, 24.93, 100.93, -121.07, -93.07, -76.07, -40.07, 129.93, 111.93, 61.93, 56.93),
                     lora_z=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                     ):

        # Gathering data in correct format
        core = np.array([core_x, core_y, 0.])
        positions = list(zip(lora_x, lora_y, lora_z))
        pos = np.array(positions)
        dat = np.array(particles)

        # Checking for zeros
        idx = np.argwhere(dat > 0.).ravel()

        data = dat[idx]
        err = np.sqrt(data)
       # print(pos.shape);
        lpos = pos[idx]

        az = azimuth # NO LOFAR convetion needed here
        zen = zenith

        d = get_distance(lpos, core_x, core_y, az, zen)
        # Fitting
        fitargs = (data, err, lpos, core_x, core_y, az, zen)
        initguess = [1e7, 30.]  # inital guess
        q, cov, infdict, ms, ierr = opt.leastsq(lora_error_function, initguess, args=fitargs, full_output=True)

        nfev = infdict['nfev']

        # Calculating chi2
        ef = lora_error_function(q, data, err, lpos, core_x, core_y, az, zen)
        chi = np.sum(ef ** 2)
        ndof = ef.shape[0] - 2
        red_chi = chi / ndof

        return q, chi, red_chi, ndof, d, data, lpos, nfev

    refitting_lora = fit_lora_ldf(new_core_x, new_core_y, azimuth, zenith, particle_densities)

    N_ch = refitting_lora[0][0]
    R_M = refitting_lora[0][1]
    S_s = 1.7

    # ------------------------------
    # SAVING results
    # ------------------------------

    # Saving core separately

    ldf_fit_core = np.array([new_core[0], new_core[1], 0.])
    #ldf_fit_core = np.array([new_core_x,new_core_y,0])
    # Save all fit parameters to output dict
    ldf_fit_output = {  'fit_parameters': p,
                        'fit_parameter_uncertainties': np.array(([np.std(p0_hist), np.std(p1_hist), np.std(p2_hist), np.std(p3_hist), np.std(p4_hist), np.std(p5_hist)])),
                        'fit_parameter_names': ['X_{+}', 'Y_{+}', '\sigma_{+}', 'A_{+},', 'X_{-}', 'A_{-}'],
                        'new_core_vxb': np.array(([out_core_x, out_core_y, 0.])),
                        'chi_2_ldf': chi2 ,
                        'red_chi_2': red_chi2 ,
                        'ndof': ndof,
                        'nfev': nfev,
                        'lora_fit_results': refitting_lora,
                        'n_ch': N_ch,
                        'r_m': R_M,
                        'remove_outliers': flag_outliers,
                        'additional_uncertainty': additional_uncertainty,
                              }

    ldf_fit_energy = p[3]/(np.sin(get_angle_to_magnetic_field_vector(zenith, azimuth)))**2
    ldf_fit_energy = ldf_fit_energy ** energy_conversion[1] * 10**energy_conversion[0]

    ldf_fit_xmax = xmax_conversion[0] + xmax_conversion[1]*p[2] + xmax_conversion[2]*p[2]**2
    ldf_fit_xmax *= -1.
    ldf_fit_xmax += full_atmosphere/np.cos(zenith)

    # Checking fit quality of LDF fit, will be refined when updating analysis
    ldf_fit_quality = True
    if red_chi2 > 3.:
        ldf_fit_quality = False
    if p[2] > 300:
        ldf_fit_quality = False
    if ((new_core_x > 1000)|(new_core_y > 1000)):
        ldf_fit_quality = False

    # Recalculate energy from LORA with geometry

    temp = np.exp(np.log(N_ch) + full_atmosphere/lora_lambda * (1/np.cos(zenith)-1/np.cos(21/180.*np.pi)))
    ldf_fit_energy_particle = 10**(lora_energy_constants[0] + lora_energy_constants[1]*np.log10(temp))*10**9

    # ---------------------------------------------
    # PLOTTING -----------------------------------------------------------
    # ---------------------------------------------

    # Making grid for contour plot
    X = np.arange(-500., 500., 10)
    Y = np.arange(-500., 500., 10)
    X, Y = np.meshgrid(X, Y)

    simple_z_grid = simple_gauss(p1, X, Y)
    ldf_z = LDF(p, X, Y)
    ldf_f = LDF(p, pos_uvw[:, 0], pos_uvw[:, 1])

    # Making large canvas for plot
    fd, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax6, ax8)) = plt.subplots(2, 4, figsize=(20, 10))

    v_max = np.max([np.max(total_p), np.max(ldf_z)])
    v_min = np.min([np.min(total_p), np.min(ldf_z)])

    # PLOT: Initial Gaussian fit for checks
    plt.subplot(2, 4, 1)
    plt.imshow(simple_z_grid, origin='lower', extent=[-500, 500, -500, 500], cmap='gnuplot2_r', vmin=v_min, vmax=v_max)
    plt.scatter(pos_uvw[:, 0], pos_uvw[:, 1], c=total_p, s=100, cmap='gnuplot2_r', vmin=v_min, vmax=v_max)
    plt.title("Initial Fit")
    plt.xlabel('vxB [m]')
    plt.ylabel('vxvxB [m]')
    # Add LORA detectors:
    lpos = get_pos_shower_plane(refitting_lora[6], particle_core[0], particle_core[1], zenith, azimuth)
    lsignal = refitting_lora[5]
    plt.scatter(lpos[:, 0], lpos[:, 1], c='#730909', s=np.log10(lsignal)*30, marker='p')

    # PLOT: Final fit
    plt.subplot(2, 4, 2)
    plt.imshow(ldf_z, origin='lower', extent=[-500, 500, -500, 500], cmap='gnuplot2_r', vmin=v_min, vmax=v_max)
    plt.scatter(pos_uvw[:, 0], pos_uvw[:, 1], c=total_p, s=100, cmap='gnuplot2_r', vmin=v_min, vmax=v_max)
    plt.scatter(0, 0, marker='+', c='k')
    plt.scatter(out_core_x, out_core_y, marker='s', c='w')
    #plt.scatter(out_core_x_un, out_core_x_un, marker='s', c='b')
    plt.xlabel('vxB [m]')
    plt.ylabel('vxvxB [m]')
    plt.title("Final Fit")

    # PLOT: Fixing ranges of axes
    mx_y = np.max(total_p)*1.2
    ml_y = np.min(total_p)*0.8

    # PLOT: Signal cuts through axes
    plt.subplot(2, 4, 5)
    plt.errorbar(pos_uvw[:, 0], total_p, uncer_noise, linestyle="None", c='b', marker='o', zorder=1)
    plt.scatter(pos_uvw[:, 0], ldf_f, c='r', zorder=2)
    if f_residuals[to_flag] != [] and flag_outliers:
        plt.scatter(flagged_pos_uvw[:, 0], flagged_total_p, c='k', marker='x')
    plt.ylim(ml_y, mx_y)
    plt.xlabel("vxB [m]")
    plt.ylabel(r'Integrated Signal [J/m$^2$]')

    plt.subplot(2, 4, 6)
    plt.errorbar(pos_uvw[:, 1], total_p, uncer_noise, linestyle="None", c='b', marker='o', zorder=1)
    plt.scatter(pos_uvw[:, 1], ldf_f, c='r', zorder=2)
    if f_residuals[to_flag] != [] and flag_outliers:
        plt.scatter(flagged_pos_uvw[:, 1], flagged_total_p, c='k', marker='x')
    plt.ylim(ml_y, mx_y)
    plt.xlabel("vxvxB [m]")
    plt.ylabel(r'Integrated Signal [J/m$^2$]')

    # Shifting to new core position
    new_positions = np.copy(pos_uvw)
    new_positions[:, 0] -= p[0] - (28.58-7.88*np.sin(azimuth))
    new_positions[:, 1] -= p[1]
    dist = np.sqrt(new_positions[:, 0]**2+new_positions[:, 1]**2)

    # PLOT: New LDF plot
    plt.subplot(2, 4, 7)
    plt.errorbar(dist, total_p, uncer_noise, linestyle="None", c='b', marker='o', zorder=1)
    plt.scatter(dist, ldf_f, c='r', zorder=2)
    plt.ylim(ml_y, mx_y)
    plt.xlabel("Distance to shower axis (radio core) [m]")
    plt.ylabel(r'Integrated Signal [J/m$^2$]')

    # Calculating deviations as function of uncertainty
    dev = (total_p-ldf_f)/uncer_noise

    # PLOT: deviations as function to identify outliers
    plt.subplot(2, 4, 3)
    plt.scatter(dist, dev)
    plt.xlabel("Distance to shower axis (radio core) [m]")
    plt.ylabel(r'Deviation (Data - Fit) [$\sigma$]')

    # PLotting LORA
    plot_par_lora = [N_ch, R_M, S_s]
    r_lora = np.arange(500)

    plt.subplot(2, 4, 4)
    plt.plot(lora_fit_function(plot_par_lora, r_lora), color='r', linewidth=2)
    plt.errorbar(refitting_lora[4], refitting_lora[5], yerr=np.sqrt(refitting_lora[5]), marker='o', linestyle=' ', color='k')
    plt.ylim(np.min(refitting_lora[5])*0.8, np.max(refitting_lora[5])*1.2)
    plt.xlim(0, np.max(refitting_lora[4])*1.2)
    plt.ylabel(r'Particle density [$m^{-2}$]')
    plt.xlabel("Distance to shower axis (radio core) )[m]")
    plt.yscale('log')

    # PLOT: Results text box
    n_out_p = 13
    ax8.text(0.05, 1-1.*1/n_out_p, 'Event: {0}'.format(eventid))
    ax8.text(0.05, 1-2.*1/n_out_p, 'Zenith: {0:.2f} deg '.format(np.degrees(zenith)))
    ax8.text(0.05, 1-3.*1/n_out_p, 'Azimuth: {0:.2f} deg '.format(np.degrees(azimuth)))
    ax8.text(0.05, 1-4.*1/n_out_p, 'Shift (wrt LORA shower): ({0:.2f}, {1:.2f}) m'.format(out_core_x, out_core_y))
    ax8.text(0.05, 1-5.*1/n_out_p, 'Shift (wrt LORA ground): ({0:.2f}, {1:.2f}) m'.format(new_core_x-particle_core[0], new_core_y-particle_core[1]))
    ax8.text(0.05, 1-6.*1/n_out_p, 'Sigma: {0:.2f} m '.format(p[2]))
    ax8.text(0.05, 1-7.*1/n_out_p, 'Scaling: {0:.2f}'.format(p[3]))
    ax8.text(0.05, 1-8.*1/n_out_p, 'Scaling ratio: {0:.2f} (0.29)'.format(p[5]/p[3]))
    ax8.text(0.05, 1-9.*1/n_out_p, 'X Offset: {0:.2f} m ([0,-105])'.format(p[4]))
    ax8.text(0.05, 1-10.*1/n_out_p, r'$\chi^2$ LOFAR: {0:.2f} ({1:.2f},{2})'.format(red_chi2, chi2, ndof))
    ax8.text(0.05, 1-11.*1/n_out_p, r'$\chi^2$ LORA: {0:.2f} ({1:.2f},{2})'.format(refitting_lora[2], refitting_lora[1], refitting_lora[3]))
    ax8.text(0.05, 1-12.*1/n_out_p, 'Function calls: {0} (LOFAR) {1} (LORA) {2} '.format(nfev, refitting_lora[7], redo_fit))

    ax8.set_xticks([])
    ax8.set_yticks([])

    try:
        # Old version of matplotlib does not know this yet
        plt.tight_layout()
    except:
        pass

    if save_plots:

        plotname = plot_prefix + "ldf_diagnose_{0}.{1}".format(eventid, plot_type)
        plt.savefig(plotname)
        plotlist.append(plotname)

    if  debug:

        plt.show()

    return ldf_fit_output, ldf_fit_core, ldf_fit_energy, ldf_fit_energy_particle, ldf_fit_xmax, ldf_fit_quality
