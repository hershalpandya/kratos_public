import os
import logging
import argparse
from datetime import datetime

# This script will 
# - find a LORA event based on a timestamp or event number
# - process LORA data, including timing calibration and determining charge deposit per scintillator
# - perform physics functions such as core fitting, energy fitting, direction reconstruction
# - generate the .json file that is necessary to start the LOFAR pipeline
