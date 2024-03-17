The software documentation
==========================

To document the software, we use ``sphinx-apidoc``, a tool for automatic generation of Sphinx sources that uses the
``sphinx-autodoc`` extension to detect and document all modules present in the source code. For more information,
please refer to the corresponding `Sphinx documentation <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_.

Automatic generation of docs on Gitlab
--------------------------------------

On the Gitlab repository, a CI/CD pipeline has been set up to automatically regenerate the documentation every time
a commit gets pushed on the main branch. This is done by using the standard Gitlab CI/CD tool, which uses Docker runners
to execute the code. The documentation is published on the
`Gitlab pages <https://hpandya.pages.science.ru.nl/kratos/index.html>`_.

In our setup, the pipeline works in two steps:

#. Generating the documentation and corresponding HTML files.
#. Pushing the HTML files to the public `pages` location.

For the Docker instance, we use ``python:buster`` as an image. This is a Debian based image, with the latest Python
installed. To install our package and extract the documentation using Sphinx, we install a virtual environment. At this
stage, it is also important to install the ``libhdf5-dev`` package using apt-get (this is the reason why we need a
Debian based image, a pure Python image would fail because HDF5 needs to be installed on the system). After installing
our package to the virtual environment, we can generate the docs.

Contributing documentation
--------------------------

Currently we use the RST format for docstrings. A good source to learn more about this is
https://sublime-and-sphinx-guide.readthedocs.io/en/latest/index.html. Some examples can also be found on the `Sphinx
website <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_.

A typical docstring has the following format:

.. code-block:: python

    def my_function(param1):
    """
    Description of the function.

    :param param1: Description of what the parameter is.
    :type param1: The expected type of the paramter.
    :return: What the function returns.
    """

Please note the whitespace between the description and the parameters. If this is not present, the docstring will not be
formatted correctly.

At some point in the future we might activate the Numpy style for docstrings. Please file an issue if you feel this is
a priority.
