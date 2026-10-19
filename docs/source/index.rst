grstbx documentation
====================

GRS toolbox
-----------

**grstbx** gathers the scientific code used to visualize and post-process the images produced by
the `GRS processor <https://github.com/Tristanovsk/grs>`__ (Glint Removal for Sentinel-2-like
sensors):

- **L2A** products: remote-sensing reflectance :math:`R_{rs}(\lambda)` corrected for the atmosphere and
  the sunglint (GRS multispectral, hGRS hyperspectral for EnMAP and PRISMA),
- **L2B** products: water-quality parameters (chlorophyll-a, suspended matter, ...) retrieved from
  :math:`R_{rs}`.

It provides:

- the selection of the products stored in a data lake by tile, date and cloud coverage
  (:class:`~grstbx.datalake.SelectFiles`),
- the loading of image series as lazy, dask-backed multi-temporal :mod:`xarray` datacubes, from
  NetCDF files or single-resolution / multiscale (pyramid) Zarr stores
  (:class:`~grstbx.driver.L2grs`),
- the decoding of the GRS bitmask ``flags`` into boolean masks (:class:`~grstbx.masking.Masking`),
- the cropping of L2A products to an area of interest and the export of the subsets as GRS products
  (NetCDF, Zarr) or GeoTIFF (:mod:`grstbx.subset`),
- interactive viewers for Jupyter notebooks, based on holoviews, datashader and panel, to browse
  the images by date and band and to draw areas or points of interest (:mod:`grstbx.visual`), and a
  web dashboard to explore L2A and L2B products and time series without notebook (``grstbx-viewer``,
  :mod:`grstbx.app`),
- matchups between satellite and in situ data (:mod:`grstbx.matchup`), and miscellaneous helpers:
  bounding boxes, AERONET-OC readers, solar irradiance, DEM-based illumination, plotting
  (:mod:`grstbx.utils`).

Install it from `PyPI <https://pypi.org/project/grstbx/>`__ (see :doc:`installation`):

.. code-block:: bash

   pip install grstbx                 # library, viewers and web dashboard (grstbx-viewer)
   pip install "grstbx[notebook]"     # + JupyterLab environment to use the viewers in notebooks

Quick start:

.. code-block:: python

   import grstbx

   dc = grstbx.L2grs(files)                    # list of GRS L2A products
   dc.get_l2a_datacube(subset=aoi)             # aoi: geopandas.GeoDataFrame
   water = grstbx.Masking(dc.datacube).get_mask(ndwi=False)
   Rrs = dc.datacube.Rrs.where(water)

   viewer = grstbx.visual.ViewSpectral(Rrs, reproject=True)   # interactive viewer, in Jupyter
   viewer.visu()

or, without notebook, the web dashboard (see :doc:`tutorials/dashboard`):

.. code-block:: bash

   grstbx-viewer /path/to/product.zarr              # or: grstbx-viewer --root /data/satellite

.. figure:: ../../illustration/grstbx_visual_tool.gif
   :alt: animated dashboard of grstbx

   Interactive exploration of a Sentinel-2 L2A time series with :class:`grstbx.visual.ViewSpectral`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   products
   processing_chain

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/l2a_image
   tutorials/dashboard
   tutorials/l2b_datacube
   tutorials/subset_export
   tutorials/utilities
   tutorials/validation
   tutorials/case_studies
   tutorials/hgrs_products

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   history
   roadmap

Related projects
----------------

- `GRS <https://github.com/Tristanovsk/grs>`__: atmospheric and sunglint correction producing the
  L2A images,
- `hGRS <https://github.com/Tristanovsk/hgrs>`__: hyperspectral version of GRS for EnMAP and PRISMA,
- `GRSdriver <https://github.com/Tristanovsk/GRSdriver>`__: readers of the L1C images used by GRS.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
