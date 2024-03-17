KRATOS User Manual
===================================

FINAL NOTES:
------------

* Documentation in HTML
    * https://hpandya.pages.science.ru.nl/kratos/ 
    * To change this, in Gitlab go to the project KRATOS and in left panel , under Settings > Pages .


STEPS BEFORE YOU START TO CONTRIBUTE:
-------------------------------------
#. Clone the repository
    * Add public ssh key in your gitlab
    * git clone using ssh 
    * ``git clone git@gitlab.science.ru.nl:hpandya/kratos.git``

#. Make a new Virtual Env so as to test pip installation and not mess up your existing working python env
    * ``python3 -m venv kratos_venv``
    * ``source kratos_venv/bin/activate``

#. From within your development virtual env, pip install the kratos package:
    * ``pip install path/to/kratos/``


#. After any git pull, if the library imports in __init__.py have changed:
    * ``pip install --upgrade path/to/kratos/``


MODULE IMPORT RULES:
---------------------
* Module imports across packages:
    * Write import statements for modules in package_*/__init__.py .
    * For example:
        * If data_io/__init__.py is empty:
            * In event/event.py we write
                * ``from ..dataio.db_manager import DB_Manager``

        * If data_io/__init__.py has: from db_manager import DB_Manager
            * In event/event.py we write
                * ``from ..dataio import DB_Manager``


* Module imports within a package, across files:
    * In data_io/db_manager, you want to import evt_dict_template() from data_io/event_dictionary:
        * ``from kratos.data_io.event_dictionary import evt_dict_template``


LOGGER USAGE RULES:
-------------------

* At the top of the file, add:
    * ``import logging``
    * ``import os``
    * ``logger = logging.getLogger(os.path.basename(__file__))``

* In the script then use one the following:
    * ``logger.warning("warning msg")``
    * ``logger.info("info msg")``
    * ``logger.error("error msg")``
    * ``logger.debug("debug msg")``

* Currently, log will be printed on screen and at the level of debug or higher. To change this, go to kratos/__init__.py and change there.

* Ref: https://docs.python.org/3/howto/logging.html#logging-from-multiple-modules

GETTING PATH TO STORAGE DIRECTORIES:
------------------------------------

* This will work only on coma:
    * ``from kratos.data_io import PathFinder``
    * ``paths = PathFinder()``
    * ``tbb_path = path_finder.get_radio_TBB_files_root_dir()``

* Should in principle be included for T2B as well. You can update data_io/storage_paths.py to do that.

* For other machines:
    * ``from kratos.data_io import PathFinder``
    * ``paths = PathFinder()``
    * ``path_finder.set_radio_TBB_files_root_dir("my_manually_entered_path")``
    * ``tbb_path = path_finder.set_radio_TBB_files_root_dir("my_manually_entered_path")``
    * Seems counter productive to first give it and then ask for it. But its important for later use in the code. A later code might just use ``path_finder.get_()`` function and if you don't set it at the first instance then the following all the calls will fail.




