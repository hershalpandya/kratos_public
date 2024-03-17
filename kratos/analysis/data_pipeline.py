# The Data pipeline will
# - find radio data files for a given event
# - read raw traces, populating Station, Antenna, Dipole objects
# - do RFI cleaning, producing cleaned traces
# - do timing calibration ("cable delays" i.e. dipole_calibration_delay)
# - do Galaxy (or other) gain calibration
# - produce calibrated traces
# - write traces & info to hdf5 and json, respectively

import os
import logging
import argparse
from datetime import datetime

from kratos.event import Event
from kratos.data_io import storage_paths, db_manager
from kratos.filter import BandpassFilter, HalfHannFilter

path_finder = storage_paths.PathFinder()


def save_plot(fig_, filename_):
    outpath = os.path.join(path_finder.get_event_plot_files_root_dir(), filename_)
    fig_.savefig(outpath, dpi=200, bbox_inches="tight")


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(1)

# Read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--event_id", type=int, help="event ID", default=94132843)
parser.add_argument("-s", "--plot_station", type=str,
                    help="station name for plotting",
                    default="CS003")
parser.add_argument("-a", "--plot_antenna", type=str,
                    help="antenna id for plotting",
                    default="a003000000")
parser.add_argument("-p", "--show-plots", action='store_true',
                    help="Show the figures on screen if set",
                    default=False)
parser.add_argument("-h5", "--load_from_h5", action='store_true',
                    help="Load Event from pre-existing hdf5 file.",
                    default=False)

args = parser.parse_args()

plot = bool(args.show_plots)

if not plot:
    logger.info("Not plotting on screen")
    import matplotlib

    matplotlib.use("Agg")  # has to go first before importing pyplot...
else:
    logger.info("Plotting results on screen")

import matplotlib.pyplot as plt

plt.set_loglevel("warning")  # why needed here??

logger.info(f"Processing event {args.event_id}")  # 92380604

# Make Event object, load station data and raw traces
this_event = Event(pathfinder=path_finder, event_id=args.event_id)

if args.load_from_h5:
    logger.debug("Loading data from pre-existing hdf5 file.")
    logger.debug("The file can have only raw traces/ not even that.")
    logger.debug("So you can basically load and save hdf5 files at different stages.")
    this_event.load_event_from_h5()
else:
    logger.debug("Loading LOFAR station (antenna) data")
    this_event.load_LOFAR_stations()

    logger.debug("Loading LOFAR raw traces")
    this_event.load_LOFAR_traces()

    # HACK
    this_event.stations = [this_event.stations[0]]
    print('Restriction to only station %s' % this_event.stations[0].get_station_name())
    # END HACK

    logger.debug("Do RFI Cleaning")
    this_event.do_rfi_cleaning()

    logger.debug("Apply bandpass filter and half-Hann window")
    my_bandpass_filter = BandpassFilter()  # use default settings for freq and roll width
    this_event.apply_filter(my_bandpass_filter, trace_type='cleaned')  # apply to and save in cleaned traces

    my_hann_filter = HalfHannFilter()  # use default half percentage
    this_event.apply_filter(my_hann_filter, trace_type='cleaned')

    logger.debug("Applying Galactic calibration")
    this_event.do_galactic_calibration("LOFAR_LBA", cal_type='old')

    logger.debug("Applying cable delays")
    this_event.do_apply_calibration_delays()

    #logger.debug("Do Antenna Unfolding")
    #this_event.do_antenna_unfolding()

    logger.debug("Save json file..")
    this_event.save_event_to_database()

    logger.debug("Save hdf5 file...")
    this_event.save_event_to_h5()

# -----------------
# MAKE PLOTS.
# -----------------

logger.debug(f"Plotting traces from station {args.plot_station}")
this_station = this_event.get_station_by_name(args.plot_station)

# -----------------
# Plot the traces
# -----------------

fig = plt.figure(figsize=(18, 6))
ax1, ax2, ax3 = fig.subplots(1, 3)

for pol in [0, 1]:
    ax1.plot(this_station.get_time_ns_array_for_trace(),
             this_event.get_dipole_trace(args.plot_station, args.plot_antenna, pol, 'raw'),
             label=f"Pol {pol}")

    ax2.plot(this_station.get_time_ns_array_for_trace(),
             this_event.get_dipole_trace(args.plot_station, args.plot_antenna, pol, 'cleaned'),
             label=f"Pol {pol}")

    ax3.plot(this_station.get_time_ns_array_for_trace(),
             this_event.get_dipole_trace(args.plot_station, args.plot_antenna, pol, 'calibrated'),
             label=f"Pol {pol}")

ax1.set_title("Raw traces")
ax1.set_ylabel("ADC Counts")
ax2.set_title("RFI cleaned traces")
ax2.set_ylabel("ADC Counts")
ax3.set_title("Calibrated traces")
ax3.set_ylabel("Voltage")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Time [ ns ]")
    ax.grid()
    ax.legend()

fig.suptitle(f"Station {args.plot_station}")
save_plot(fig, f"CalibratedTraces_{args.event_id}_{args.plot_station}.png")

# -----------------
# Plot the spectra
# -----------------
fig2 = plt.figure(figsize=(18, 6))
ax1, ax2, ax3 = fig2.subplots(1, 3)

ax1.plot(
    this_station.get_fftfreq_MHz()[1:-1],
    this_station.get_raw_avg_powerspectrum()[1:],
    c="r",
    label="flagged as RFI",
)

ax1.plot(
    this_station.get_fftfreq_MHz()[1:-1],
    this_station.get_cleaned_avg_powerspectrum()[1:]
    , c="b"
)
ax1.set_title("RFI flags")

for pol, ax in [[0, ax2], [1, ax3]]:
    ax.plot(
        this_station.get_fftfreq_MHz(),
        this_event.get_dipole_spectrum(args.plot_station, args.plot_antenna, pol, 'raw'),
        c="grey", label=f"Raw spectrum pol:{pol}")

    ax.plot(
        this_station.get_fftfreq_MHz(),
        this_event.get_dipole_spectrum(args.plot_station, args.plot_antenna, pol, 'calibrated'),
        c="purple", label=f"Calibrated spectrum pol:{pol}",
    )

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Frequency [ MHz ]")
    ax.set_ylabel("Power spectrum [ a.u. ]")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

fig2.suptitle(f"Station {args.plot_station}")
save_plot(fig2, f"CalibratedSpectra_{args.event_id}_{args.plot_station}.png")

logger.info(f"Figures saved to {path_finder.get_event_plot_files_root_dir()}")
if plot:
    fig.show()
    fig2.show()

db = db_manager.DBManager()
event_dict = db.get_event_dictionary_from_json_file(args.event_id)

now = datetime.now()
now_string = now.strftime("%B %d, %Y, %H:%M:%S")

event_dict["status"]["processed"] = True
event_dict["status"]["last_processed"] = now_string

# db.write_event_dictionary_to_disk(event_dict)
logger.info("Cannot write event dict to disk when a new (sub)key has been added")

logger.info("Done!")
