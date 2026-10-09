grstbx documentation
====================

GRS toolbox
-----------

**grstbx** gathers the scientific code used to visualize and post-process the images produced by
the `GRS processor <https://github.com/Tristanovsk/grs>`__ (Glint Removal for Sentinel-2-like
sensors):

- **L2A** products: remote-sensing reflectance :math:`R_{rs}(\lambda)` corrected for the
  atmosphere and the sunglint,
- **L2B** products: water quality parameters (chlorophyll-a, suspended matter, ...) retrieved from
  :math:`R_{rs}`.

It provides:

- the selection of the products stored in a data lake by tile, date and cloud coverage
  (:class:`~grstbx.datalake.SelectFiles`),
- the loading of image series as lazy, dask-backed multi-temporal :mod:`xarray` datacubes, from
  netCDF files or single-resolution / multiscale (pyramid) Zarr stores
  (:class:`~grstbx.driver.L2grs`),
- the decoding of the GRS bitmask ``flags`` into boolean masks (:class:`~grstbx.masking.Masking`),
- interactive viewers for Jupyter notebooks, based on holoviews, datashader and panel, to browse
  the images by date and band and to draw areas or points of interest (:mod:`grstbx.visual`),
- miscellaneous helpers: bounding boxes, AERONET-OC readers, solar irradiance, DEM-based
  illumination, plotting (:mod:`grstbx.utils`).

.. code-block:: python

   import grstbx

   dc = grstbx.L2grs(files)
   dc.get_l2a_datacube(subset=aoi)          # aoi: geopandas.GeoDataFrame
   water = grstbx.Masking(dc.datacube).get_mask(ndwi=False)
   Rrs = dc.datacube.Rrs.where(water)

.. figure:: ../../illustration/grstbx_visual_tool.gif
   :alt: animated dashboard of grstbx

   Interactive exploration of a Sentinel-2 L2A time series with :class:`grstbx.visual.ViewSpectral`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   products

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/select_files
   tutorials/datacube
   tutorials/masking
   tutorials/visualization
   tutorials/l2a_image
   tutorials/utilities
   tutorials/validation
   tutorials/case_studies

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   history

Related projects
----------------

- `GRS <https://github.com/Tristanovsk/grs>`__: atmospheric and sunglint correction producing the
  L2A images,
- `GRSdriver <https://github.com/Tristanovsk/GRSdriver>`__: readers of the L1C images used by GRS.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
