Installation
============

grstbx requires Python >= 3.11. It is published on `PyPI <https://pypi.org/project/grstbx/>`__.

From PyPI
---------

.. code-block:: bash

   pip install grstbx                 # library, viewers and web dashboard (grstbx-viewer)
   pip install "grstbx[notebook]"     # + JupyterLab environment to use the viewers in notebooks

All the dependencies except ``xesmf`` are available as wheels on PyPI, so that grstbx can be installed
in a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install "grstbx[notebook]"

Conda environment
-----------------

The geospatial libraries (GDAL, PROJ, GEOS) and ``xesmf`` (ESMF) are compiled libraries, easier to
install from conda-forge. In a conda environment, install them first, then grstbx with pip:

.. code-block:: bash

   conda create -n grstbx -c conda-forge python=3.12 gdal rasterio pyproj cartopy netcdf4
   conda activate grstbx
   pip install "grstbx[notebook]"

From source
-----------

To run the notebooks of `the repository <https://github.com/Tristanovsk/grstbx>`__ or to develop
grstbx, clone it and create the conda environment of ``environment.yml``, which installs all the
dependencies (including Jupyter and ``xesmf``) and grstbx in editable mode:

.. code-block:: bash

   git clone https://github.com/Tristanovsk/grstbx.git
   cd grstbx
   conda env create -f environment.yml
   conda activate grstbx

or install the clone with pip in an existing environment:

.. code-block:: bash

   pip install ".[notebook]"
   pip install -e ".[notebook]"      # editable install, for development

``requirements.txt`` lists the same dependencies as ``pip install ".[notebook]"``
(``pyproject.toml`` is the reference).

Options
-------

``pip install grstbx`` installs the library, its interactive viewers and the web dashboard
``grstbx-viewer`` (see :doc:`tutorials/dashboard`). The optional dependencies are grouped in extras,
e.g. ``pip install "grstbx[notebook,regrid]"``:

==============  ==============================================================  ======================================
extra           content                                                         usage
==============  ==============================================================  ======================================
``notebook``    JupyterLab, ipykernel, jupyter_bokeh                            run the notebooks and the viewers
``regrid``      xesmf                                                           :meth:`grstbx.utils.Reproj.regridding`
``docs``        Sphinx, sphinx-book-theme, myst-nb...                           build this documentation
``all``         ``notebook`` and ``docs``
==============  ==============================================================  ======================================

``xesmf`` depends on ESMF, which is not available on PyPI: install it with
``conda install -c conda-forge xesmf``.

Export of subsets
~~~~~~~~~~~~~~~~~

:func:`grstbx.export_l2a <grstbx.subset.export_l2a>` and
:func:`grstbx.crop_and_export <grstbx.subset.crop_and_export>` write the subsets with the exporters
of the `GRS processor <https://github.com/Tristanovsk/grs>`__: they need the ``grs`` package >= 3.0.1
in the same environment. It is imported only when exporting; the rest of grstbx (including
:func:`~grstbx.subset.crop_l2a` and :func:`~grstbx.subset.export_rrs_geotiff`) works without it.

Jupyter kernel
--------------

The interactive viewers (:mod:`grstbx.visual`) are meant to be used in Jupyter. To register the
environment as a kernel:

.. code-block:: bash

   python -m ipykernel install --user --name=grstbx

Check the installation
----------------------

.. code-block:: bash

   python -c "import grstbx; print(grstbx.__version__)"
   grstbx-viewer --help
   pip check

Troubleshooting
~~~~~~~~~~~~~~~

``PROJ: proj.db contains DATABASE.LAYOUT.VERSION.MINOR = ... It comes from another PROJ installation``
   The environment variables ``PROJ_DATA`` / ``PROJ_LIB`` (or ``GDAL_DATA``) point to the data of
   another PROJ installation, typically a conda environment activated before a pip virtual
   environment. Unset them (``unset PROJ_DATA PROJ_LIB GDAL_DATA``) or install everything in the
   same conda environment.

Building the documentation
--------------------------

From a clone of the repository:

.. code-block:: bash

   pip install ".[docs]"
   cd docs
   make html     # output in docs/build/html
