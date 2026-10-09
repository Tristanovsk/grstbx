Installation
============

grstbx requires Python >= 3.11. It is installed from a local copy of
`the repository <https://github.com/Tristanovsk/grstbx>`__:

.. code-block:: bash

   git clone https://github.com/Tristanovsk/grstbx.git
   cd grstbx

Conda environment (recommended)
-------------------------------

The geospatial libraries (GDAL, PROJ, GEOS) and ``xesmf`` (ESMF) are compiled libraries, easier to
install from conda-forge. ``environment.yml`` creates an environment ``grstbx`` with all the
dependencies (including Jupyter and ``xesmf``) and installs grstbx in editable mode:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate grstbx

To install grstbx in an existing conda environment, install the compiled libraries first, then the
package with pip:

.. code-block:: bash

   conda install -c conda-forge gdal rasterio pyproj cartopy netcdf4
   pip install ".[notebook]"

pip only
--------

All the dependencies except ``xesmf`` are available as wheels on PyPI:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install ".[notebook]"

Options
-------

``pip install .`` installs the library and its interactive viewers. The optional dependencies are
grouped in extras:

==============  ==============================================================  ======================================
extra           content                                                         usage
==============  ==============================================================  ======================================
``notebook``    JupyterLab, ipykernel, jupyter_bokeh                            run the notebooks and the viewers
``regrid``      xesmf                                                           :meth:`grstbx.utils.Reproj.regridding`
``docs``        Sphinx, sphinx-book-theme, myst-nb...                           build this documentation
``all``         ``notebook`` and ``docs``
==============  ==============================================================  ======================================

.. code-block:: bash

   pip install ".[notebook,docs]"
   pip install -e ".[notebook]"      # editable install, for development

``requirements.txt`` lists the same dependencies as ``pip install ".[notebook]"``
(``pyproject.toml`` is the reference). ``xesmf`` depends on ESMF, which is not available on PyPI:
install it with ``conda install -c conda-forge xesmf``.

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

.. code-block:: bash

   pip install ".[docs]"
   cd docs
   make html     # output in docs/build/html
