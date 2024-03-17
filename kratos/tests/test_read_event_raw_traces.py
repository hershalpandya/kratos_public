from kratos.event import event
from kratos.station import station
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(os.path.basename(__file__))

tbb_file_directory = "/vol/astro3/lofar/vhecr/lora_triggered/data/"
my_event = event.Event(tbb_storage_path=tbb_file_directory)
# bypass event loading for now
tbb_filename = ["L78862_D20121205T051644.039Z_CS002_R000_tbb.h5"]
full_tbb_filename = [os.path.join(tbb_file_directory, tbb_filename[0])]

my_station = station.Station(
    data_filename=full_tbb_filename
)  # have directory in Event, pass it down
my_event.stations.append(my_station)
my_station.load_LOFAR_antennas()

blocknr = 3
alignment_shift = 1000
# TODO: fix the load_LOFAR_traces() call below.
my_station.load_LOFAR_traces(blocknr, alignment_shift)

trace_even = my_station.get_dipole_for_index(6).raw_trace
trace_odd = my_station.get_dipole_for_index(7).raw_trace


plt.figure()
plt.plot(my_station.get_time_ns_array_for_trace(), trace_even, label="Pol 0")
plt.plot(my_station.get_time_ns_array_for_trace(), trace_odd, label="Pol 1")
plt.xlabel("Time [ ns ]")
plt.ylabel("Voltage [ a.u. ]")
plt.grid()

my_station.do_rfi_cleaning()

cleaned_trace_even = my_station.get_dipole_for_index(6).cleaned_trace
cleaned_trace_odd = my_station.get_dipole_for_index(7).cleaned_trace
plt.plot(my_station.get_time_ns_array_for_trace(), cleaned_trace_even, label="Cleaned Pol 0")
plt.plot(my_station.get_time_ns_array_for_trace(), cleaned_trace_odd, label="Cleaned Pol 1")
plt.legend(loc="best")

freq_axis = my_station.get_fftfreq_MHz()

freq_axis = freq_axis[0:-1]  # avg_powerspectrum has one freq channel less...
plt.figure()
cleaned_spectrum = np.copy(my_station.avg_powerspectrum)
cleaned_spectrum[my_station.dirty_channels] *= 0.0

plt.plot(freq_axis[1:], my_station.avg_powerspectrum[1:], c="r", label="flagged as RFI")
plt.plot(freq_axis[1:], cleaned_spectrum[1:], c="b")

plt.xlabel("Frequency [ MHz ]")
plt.ylabel("Power spectrum [ a.u. ]")
plt.yscale("log")
plt.legend(loc="best")
plt.grid()


plt.show()
