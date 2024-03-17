# The particle data pipeline will
# - process the daily .root events that contain LORA data
# - find events that produce a radio triggger
# - read in traces
# - do timing calibration
# - do event reconstruction (direction, core, energy)
# - write event information to json files, the first step of the CR pipeline


import os
import logging
import argparse
from datetime import datetime
import uproot

from kratos.particle.particle_event import Event


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--event_id", type=int, help="event ID", default=94132843)
parser.add_argument("-t", "--timestamp", type=int, help="UTC timestamp", default=94132843)
parser.add_argument("-d", "--date", type=int, help="YYYYMMDD", default=94132843)

parser.add_argument("-p", "--show-plots", action='store_true',
                    help="Show the figures on screen if set",
                    default=False)

# root events are saved in the format20230111_0038.root

args = parser.parse_args()

plot = bool(args.show_plots)

if not plot:
    logger.info("Not plotting on screen")
    import matplotlib

    matplotlib.use("Agg")  # has to go first before importing pyplot...
else:
    logger.info("Plotting results on screen")

import matplotlib.pyplot as plt

this_event = Event(pathfinder=path_finder, event_id=args.event_id)

