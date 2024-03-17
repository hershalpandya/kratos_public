Kosmic-ray Radio Analysis TOolS (KRATOS)
========================================

This is the successor to PyCrTools, a library to read and analyse cosmic-ray events observed with radio telescopes.

Authors
-------

#. Hershal Pandya, hershalpandya@gmail.com .
#. Katie Mulrey, K.Mulrey@astro.ru.nl .
#. Arthur Corstanje, a.corstanje@astro.ru.nl .
#. Mitja Desmet, Mitja.Desmet@vub.be .
#. Nikolaos Karastathis, nikolaos.karastathis@kit.edu .
#. Jhansi Bhavani V., jhansi.bhavani@ku.ac.ae .
#. Brian Hare, b.h.hare@rug.nl .

Package/Classes hierarchy
-------------------------

The data files used for analysis should be put in a local `data/` directory, located in the root directory.

* data/
    * antenna_respons_model/
    * calibration/
    * experiment_parameters.txt

Documentation files are stored in the `docs/` directory.

* docs/
    * conf.py
    * index.rst
    * readme.rst
    * unittesting.rst
    * usermanual.rst
    * documentation.rst

All the source files are in ``kratos`` folder. It contains several submodules, representing the objects useful for
analysis:

* ``dipole``
    * __init__.py
    * dipole.py
* ``antenna``
    * __init__.py
    * antenna.py
* ``station``
    * __init__.py
    * station.py
* ``event``
    * __init__.py
    * event.py

Furthermore, it has also some submodules with convenience functions for physics and accessing data:

* ``data_io``
    * __init__.py
    * db_manager.py
    * event_dictionary.py
    * find_rfi.py
    * lofar_io.py
    * lora_io.py
    * metadata.py
    * raw_TBB_io.py
    * signal_processing.py
    * storage_paths.py
    * utilities.py
* ``physics``
    * __init__.py
    * antenna_model.py
    * beamformer.py
    * coordinate_transformations.py
    * galactic_calibration.py
    * generic_physics_functions.py
    * ldf.py
    * planewave.py
    * unfold_antenna_response.py

Lastly, some subfolders contain example scripts how to use the classes and how to test them:

* analysis/
    * data_pipeline.py
    * sample_analysis.py
    * storage_locations.py
* examples/
    * how_to_simply_create_json.py
* tests/
    * test_h5_file_read.py
    * test_read_event_raw_traces.py
