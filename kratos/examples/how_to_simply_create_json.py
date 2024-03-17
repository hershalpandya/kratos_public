from kratos.data_io import DBManager
import numpy as np
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))

db = DBManager()

# will give you whatever is the current template in repository
new_dict = db.get_event_dictionary_empty()

# lets print out some stuff:
logger.info(f"{new_dict['LORA']}")

# now fill it in:
new_dict["event_id"] = 1234561
new_dict["LORA"]["energy_GeV"] = 1000000
new_dict["LORA"]["azimuth_rad"] = np.pi
new_dict["LORA"]["zenith_rad"] = np.pi / 6.0

# done with work, save to disk again:
# BE CAREFUL. THIS WILL RE-WRITE WITH A SMALL MSG. NO GOING BACK.
db.write_event_dictionary_to_disk(new_dict)
