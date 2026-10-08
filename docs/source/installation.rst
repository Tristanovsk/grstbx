Installation
============

grstbx requires Python >= 3.11.

Conda environment
-----------------

The geospatial libraries (GDAL, PROJ) are easier to install with conda:

.. code-block:: bash

   conda create -n grstbx -c conda-forge python=3.12 gdal pyproj cartopy
   conda activate grstbx

Package
-------

Clone `the repository <https://github.com/Tristanovsk/grstbx>`__ and install the package from the
local copy:

.. code-block:: bash

   git clone https://github.com/Tristanovsk/grstbx.git
   cd grstbx
   pip install .

Optional dependencies:

.. code-block:: bash

   pip install ".[regrid]"   # xesmf, for grstbx.utils.Reproj.regridding
   pip install ".[docs]"     # to build this documentation

Jupyter kernel
--------------

The interactive viewers (:mod:`grstbx.visual`) are meant to be used in Jupyter. To register the
environment as a kernel:

.. code-block:: bash

   python -m ipykernel install --user --name=grstbx

Building the documentation
--------------------------

.. code-block:: bash

   pip install ".[docs]"
   cd docs
   make html     # output in docs/build/html
