# Test script for LOFAR-CR hdf5 file read-in.
# Author: A. Corstanje, Oct 2022
# Based on B. Hare's raw_tbb_IO, and K. Mulrey's Jupyter-notebook pipeline

from data_io import raw_tbb_IO
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

radio_files = [
    "/vol/astro3/lofar/vhecr/lora_triggered/data/L78862_D20121205T051644.039Z_CS002_R000_tbb.h5"
]
TBB_data = raw_tbb_IO.MultiFile_Dal1([radio_files[0]])
block_size = 65536
print(
    "Antenna set: ", TBB_data.antennaSet
)  # raw_tbb_IO has issues with style compliance even when you're quite lenient like me
clock_frequency = TBB_data.SampleFrequency

nBlocks = np.floor(TBB_data.DataLengths[0] / block_size)
all_dipoles = [int(x) % 100 for x in TBB_data.dipoleNames]
nof_dipoles = len(all_dipoles)
raw_data = np.zeros([nof_dipoles, block_size])

shift_from_lora = 1000  # stub
block_number_lora = 3

for i in np.arange(nof_dipoles):
    raw_data[i] = TBB_data.get_data(
        block_size * (block_number_lora) + shift_from_lora, block_size, antenna_index=i
    )
    # raw_data[i] = np.roll(raw_data[i], -1*int(shift)) # do not use np.roll for this! It is circular shifting

peak_index = np.median(np.argmax(np.abs(raw_data), axis=1))  # simplistic for testing
peak_index = int(peak_index)

window_width = 1024
start = peak_index - window_width // 2
end = peak_index + window_width // 2

time_axis = 5.0 * np.arange(window_width)  # ns

dipole_index = 6
plt.figure()
plt.plot(time_axis, raw_data[dipole_index][start:end], label="even pol")
plt.plot(time_axis, raw_data[dipole_index + 1][start:end], label="odd pol")
plt.xlabel("Time [ ns ]")
plt.ylabel("Raw signal [ ADC units ]")
plt.legend(loc="best")
