Unit testing the package
========================

This package uses `pytest` to perform unit tests. The files containing the tests should go in the ``testing/`` directory (at root level). These files should match either ``test_*.py`` or ``*_test.py`` formats. After installing the module, simply running the command `pytest` will perform all tests found in the directory.

In the testing files, `pytest` looks for the following things:

* Functions or methods outside a class, prefixed with ``test``
* ``test`` prefixed functions or methods inside a class prefixed with ``Test`` (without a ``__init__`` method)

