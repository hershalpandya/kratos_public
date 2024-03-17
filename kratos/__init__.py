import sys
import logging

from . import antenna
from . import data_io
from . import dipole
from . import event
from . import physics
from . import station
from . import filter
from . import particle

logging.basicConfig(stream=sys.stdout)
