Processing Chain Description
============================

This page documents grstbx following the
`Processing Chain Documentation Template <https://processing-chain-guidelines.readthedocs.io/en/latest/Data_processing_chain_template/>`_.

Process description
-------------------

Description
~~~~~~~~~~~

grstbx is the post-processing and visualization toolbox of the products of the
`GRS processor <https://github.com/Tristanovsk/grs>`__: the L2A products (remote-sensing reflectance
:math:`R_{rs}` corrected for the atmosphere and the sunglint, GRS multispectral and hGRS hyperspectral)
and the L2B products (water-quality parameters). It is a Python library (see :doc:`api`) with three
processing chains:

- **subset extraction**: an area of interest (AOI) is cut from an L2A product and written as a new,
  smaller L2A product with the layout and packing of GRS, or as a GeoTIFF of :math:`R_{rs}`
  (:mod:`grstbx.subset`, :func:`~grstbx.subset.crop_and_export`, see :doc:`tutorials/subset_export`);
- **datacubes and matchups**: series of products are stacked into lazy, multi-temporal xarray
  datacubes (:class:`~grstbx.driver.L2grs`), masked with the pixel classification
  (:class:`~grstbx.masking.Masking`) and compared with in situ data (:mod:`grstbx.matchup`, see
  :doc:`tutorials/validation`);
- **interactive exploration**: the viewers of :mod:`grstbx.visual` in Jupyter, and the web dashboard
  ``grstbx-viewer`` (entry point ``grstbx.app:main``, see :mod:`grstbx.app` and
  :doc:`tutorials/dashboard`), a service that reads only the pixels displayed and exports the values of
  the drawn points and the statistics of the drawn polygons.

Application Domain (Granule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Subset extraction**: one run corresponds to **one L2A product (one tile or footprint, one date) and
  one AOI**, and gives one subset product. A series of products is processed by a loop over the
  products, with the same AOI.
- **Datacubes and matchups**: one run corresponds to **one site**: a series of L2A or L2B products of
  one tile (or of subsets of one site), over a period, optionally cropped to a box around the site; the
  matchups pair the images of the datacube with the in situ data of the site.
- **Interactive exploration**: one dashboard session opens **one product, a series of products in the
  same CRS (UTM zone), or one NetCDF datacube**, of one level (L2A or L2B). The products of a series can
  be on different grids (e.g. subsets of several tiles of one UTM zone); a series can be given as folders
  (e.g. one tile of the tree ``<tile>/YYYY/MM/DD/<product>`` of the GRS batches, over a period). The
  products are opened when they are displayed.

Scheduling and Triggers
~~~~~~~~~~~~~~~~~~~~~~~

grstbx has no built-in scheduler; its chains are triggered externally:

- **Interactively**: from Python or Jupyter (see the :doc:`tutorials <tutorials/l2a_image>`), or with
  the web dashboard, started manually with ``grstbx-viewer`` (see :doc:`tutorials/dashboard`).
- **Batch**: a Python loop over the products, typically listed with
  :class:`~grstbx.datalake.SelectFiles` (tile, dates, cloud coverage) or a glob pattern.
  :func:`~grstbx.subset.crop_and_export` skips the outputs that already exist (unless
  ``overwrite=True``), so that an interrupted batch can be restarted. On an HPC cluster, the script is
  submitted as a job of the cluster scheduler (e.g. SLURM ``sbatch``); no job script is shipped with
  grstbx. There is no periodicity of its own: the extraction of new products (e.g. after a GRS
  reprocessing) is delegated to the calling scripts.
- **Service**: ``grstbx-viewer`` runs until it is stopped (Ctrl-C). Started on a processing server with
  ``--address 0.0.0.0``, it is shared by the users of the network, each browser session opening its own
  products; it can be kept running by any service manager (e.g. systemd, a terminal multiplexer).

Dataflow
--------

.. mermaid:: _diagrams/dataflow.mmd

The L2A subsets are GRS products: they are inputs of the two other chains, like the full products.

Inputs
------

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 22 15 63

   * - Data Type Name
     - Cardinality
     - Selection Criteria
   * - GRS L2A product
     - 1..n (mandatory for the subset extraction and the L2A datacubes)
     - Zarr store (single resolution or multiscale pyramid with an ``ancillary`` group) or NetCDF product
       folder ``<name>/`` holding ``<name>.nc`` and ``<name>_anc.nc``; hGRS L2A NetCDF file (EnMAP,
       PRISMA). Selected by tile, date and cloud coverage (:class:`~grstbx.datalake.SelectFiles`), by
       path or by glob pattern.
   * - GRS L2B product
     - 0..n
     - Zarr store or NetCDF file of the water-quality parameters, for the L2B datacubes, the matchups and
       the dashboard.
   * - Area of interest
     - 0..1 (mandatory for the subset extraction)
     - Polygon(s) in any CRS: ``geopandas.GeoDataFrame``, vector file (GeoJSON, shapefile, GeoPackage),
       box around a point (:func:`~grstbx.subset.aoi_from_box`), WKT, or polygon drawn on the viewers.
   * - In situ data
     - 0..n
     - Measurements of the site for the matchups: e.g. HYPERNETS radiometer (NetCDF), monitoring
       networks (CSV), AERONET-OC (:meth:`~grstbx.utils.Data.read_aeronet_ocv3`).
   * - Folder of products
     - 0..n
     - Tree ``<tile>/YYYY/MM/DD/<product>`` of the GRS batches (root, tile, year or month folder), explored
       by :func:`~grstbx.app.find_products`; selection by tile, period and last processing of each
       acquisition (:func:`~grstbx.app.select_products`).
   * - NetCDF datacube
     - 0..1
     - Datacube with a ``time`` dimension saved from :class:`~grstbx.driver.L2grs`, reopened by the
       dashboard (loaded in memory when smaller than 1 GB).

Outputs
-------

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 30 15 55

   * - Data Type Name
     - Cardinality
     - Description
   * - L2A subset (Zarr)
     - 0..1 per input product
     - Subset with the layout of the GRS products: full resolution, overviews (pyramid) and ancillary
       group (see `Output product format`_). Written by :func:`~grstbx.subset.export_l2a` with the writers
       of GRS (``grs`` package >= 3.0.1).
   * - L2A subset (NetCDF)
     - 0..1 per input product
     - Alternative format: main file and ancillary file ``<name>_anc.nc``, without overviews.
   * - GeoTIFF of :math:`R_{rs}`
     - 0..1 per input product
     - Multi-band float32 GeoTIFF for GIS software (:func:`~grstbx.subset.export_rrs_geotiff`).
   * - Datacube
     - 0..1 per site
     - Lazy (dask) ``xarray.Dataset`` with a ``time`` dimension and the proportion of pixels raised by
       each flag (``flag_<name>``), in memory; saved as NetCDF by the calling script if needed.
   * - Matchup table and statistics
     - 0..1 per site and parameter
     - ``pandas.DataFrame`` of the satellite / in situ pairs (:func:`~grstbx.matchup.get_matchups`) and
       agreement statistics (:func:`~grstbx.matchup.matchup_metrics`), scatter plots
       (:func:`~grstbx.matchup.plot_matchup`).
   * - Dashboard exports
     - 0..n per session
     - CSV of the values of the drawn points and of the statistics of the drawn polygons for all the
       dates, GeoJSON of the points and polygons, downloaded by the browser.

Output product format
~~~~~~~~~~~~~~~~~~~~~

The L2A subsets have the format of the GRS products (see the
`processing chain of GRS <https://grs.readthedocs.io/en/latest/processing_chain.html>`__): the format is
chosen from the extension of the output path given to :func:`~grstbx.subset.export_l2a` (or with
``fmt=``), and :func:`~grstbx.subset.subset_path` builds it from the name of the input product and the
name of the AOI.

**Zarr store with pyramids**::

    <name>.zarr/              root group: attributes and multiscales, no data
    <name>.zarr/0/            full resolution
    <name>.zarr/1/            overview, 2 x coarser (subsets > 512 pixels)
    ...
    <name>.zarr/ancillary/    ancillary data on the coarse grid

**NetCDF**: ``<name>/<name>.nc`` (main product) and ``<name>/<name>_anc.nc`` (ancillary data), zlib
compression.

The ancillary data are the CAMS fields and the gaseous transmittance of GRS.

* **Packing**: ``Rrs``, ``BRDFg``, ``aot550``, the angles and ``dem`` are stored as scaled ``int16``
  (e.g. ``Rrs``: scale 1e-5), ``flags`` and ``mask`` unchanged: the export of a subset of a GRS product
  is lossless.
* **Area of interest**: the product is cut to the bounding box of the AOI; with ``mask_outside=True``
  (default), the pixels of the box outside the polygon(s) are flagged as missing (NaN for the float
  variables, bit 0 ``nodata`` of ``flags`` raised, ``mask`` set to 1). The ancillary data are cut to the
  box enlarged by one coarse cell.
* **Reading**: a subset is reopened with :class:`~grstbx.driver.L2grs` (or
  :func:`~grstbx.subset.open_l2a`) like any GRS product.

**GeoTIFF**: one float32 band per wavelength, named ``Rrs_<wl>``, NaN for the missing pixels
(``water_only=True``: valid water pixels only).

**Dashboard exports** (CSV):

- ``points_values.csv``: one row per point and date: ``point``, ``lon``, ``lat``, ``time``, then one
  column per wavelength (``Rrs_<wl>``, L2A) or per parameter (L2B);
- ``polygons_statistics.csv``: one row per polygon, date and statistic (``median``, ``mean``, ``std``,
  ``q05``, ``q25``, ``q75``, ``q95``, ``count``), then one column per wavelength or parameter;
- ``points.geojson``, ``polygons.geojson``: the geometries (EPSG:4326).

See :doc:`tutorials/subset_export` for the crop and export options.

Return Codes
------------

**Web dashboard** ``grstbx-viewer`` (:func:`grstbx.app.main`):

- ``0``: normal exit, after the server is stopped (Ctrl-C).
- ``1``: error at start-up, reported with its traceback (e.g. ``OSError: [Errno 98] Address already in
  use`` when the port is taken: choose another one with ``--port``).
- ``2``: invalid command-line arguments (``argparse``), with the usage message.

The errors raised while opening products in a session (missing file, L2A and L2B products mixed,
products in different CRS, no parameter to display) are shown to the user as notifications in the
browser: the server keeps running.

**Python API**: the chains are functions; their errors are raised as Python exceptions, which stop a
batch unless the calling script catches them:

- ``FileNotFoundError``: input product or glob pattern not found;
- ``ValueError``: inconsistent inputs (e.g. products of several CRS or levels opened together);
- rioxarray ``NoDataInBounds``: AOI outside the product.

Two cases are not errors and must be checked by the caller:
:func:`~grstbx.subset.crop_and_export` returns the path of the existing output without recomputing it
when ``overwrite=False``, and :meth:`L2grs.get_l2a_datacube <grstbx.driver.L2grs.get_l2a_datacube>`
(or ``get_l2b_datacube``) sets ``no_product = True`` and creates no datacube when no image is kept.

Log Format
----------

grstbx does not configure the logging. The library writes ``INFO`` records with the standard
``logging`` module while it builds the datacubes (``opening l2a image: <file>``,
``loading the subsets``, ``computing flags statistics``, ``concatenate rasters``); they are shown once
the calling script configures the logging, e.g.:

.. code-block:: python

   import logging

   log_format = '%(asctime)s %(levelname)s | %(name)s::%(funcName)s | %(message)s'
   logging.basicConfig(format=log_format, level=logging.INFO)

The dashboard prints its address on the standard output (``Launching server at
http://localhost:5006``), and the web server (bokeh, tornado) writes its warnings and errors on the
standard error with the default format of the ``logging`` module:

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 25 75

   * - Field
     - Description
   * - ``levelname``
     - Classification level: ``WARNING`` or ``ERROR`` (``INFO`` and ``DEBUG`` are not shown by default).
   * - ``name``
     - Logger name (module emitting the record, e.g. ``bokeh.server.views.ws``).
   * - ``message``
     - Free-text message body.

Format string: ``"%(levelname)s:%(name)s:%(message)s"``, e.g.
``ERROR:bokeh.server.views.ws:Refusing websocket connection from Origin 'http://myserver:5006'``
(origin not allowed: add it with ``--allow-websocket-origin``).

Required Resources
------------------

Actual needs depend on the size of the products, of the AOI and of the time series. Indicative values,
measured on a workstation (40 cores, 62 GB of RAM, local disk) with the test products of the tutorials:

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 30 30 40

   * - Resource Type
     - Quantity
     - Source / notes
   * - CPU
     - 1 process; dask threads for the Zarr stores, a single thread for the NetCDF files (HDF5 reads
       are serialized)
     - :class:`~grstbx.driver.L2grs`, exports of :mod:`grstbx.app`
   * - RAM, subset extraction
     - < 1 GB (peak 0.95 GB for a 15 x 12 km subset of a Sentinel-2 tile)
     - only the chunks intersecting the AOI are read
   * - RAM, datacubes
     - from a few GB to tens of GB: the subsets of all the images are read in parallel before being
       stacked (peak 14 GB for 95 L2B subsets of 409 x 409 pixels, datacube of 2.5 GB; about 20 GB for 22
       L2A images of 1150 x 1300 pixels and 11 bands)
     - :doc:`tutorials/l2b_datacube`, :doc:`tutorials/case_studies`; reduce the box or the period if
       needed
   * - RAM, dashboard
     - per session: the cache of the data read (``--cache-size``, 2 GB by default) and the products in
       use (peak 0.8 GB for a Sentinel-2 tile); NetCDF datacubes smaller than 1 GB are loaded in memory
     - only the pixels displayed are read; the products of a time series are opened when displayed (a
       series of 108 L2B tiles opens in 0.3 s)
   * - Execution time
     - subset extraction: 2 to 5 s per product (NetCDF, Zarr); datacube of 95 L2B subsets: 3 min;
       dashboard: 4 s to open and draw a Sentinel-2 tile, exports in 1 s for one product, about 30 s for
       95 NetCDF products, 3 to 8 s for a datacube of 716 dates
     - same measurements
   * - Disk, input
     - size of the L2A / L2B products (a Sentinel-2 L2A tile at 20 m: a few GB)
     - read only; Zarr stores can be read partially
   * - Disk, output
     - a few MB per subset (4.6 MB NetCDF, 6.7 MB Zarr for a 15 x 12 km subset at 20 m)
     - written under the output directory

Data Types
----------

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 18 32 15 35

   * - Data Name
     - Description
     - Granule
     - Nomenclature
   * - GRS L2A product
     - Remote-sensing reflectance and sunglint, Sentinel-2 or Landsat
     - one tile, one date
     - ``S2{A,B,C}_MSIL2AGRS_<datetime>_N<baseline>_R<orbit>_T<tile>_<datetime>_V<version>.zarr``
       (GRS >= 3), NetCDF folder ``<name>/`` with ``<name>.nc`` and ``<name>_anc.nc`` (GRS 2)
   * - hGRS L2A product
     - Hyperspectral remote-sensing reflectance, EnMAP or PRISMA
     - one footprint, one date
     - e.g. ``PRS_L2A_hgrs_<start>_<stop>_<n>_V<version>.nc``
   * - GRS L2B product
     - Water-quality parameters retrieved from :math:`R_{rs}`
     - one tile or subset, one date
     - e.g. ``S2{A,B}_MSIL2B_<datetime>_N<baseline>_R<orbit>_T<tile>_<datetime>.nc`` or ``.zarr``
   * - L2A subset
     - Subset of an L2A product to an AOI, GRS format
     - one AOI, one date
     - ``<odir>/<site>/<product>_<site>.zarr`` or ``<odir>/<site>/<product>_<site>/<product>_<site>.nc``
       (and ``_anc.nc``), see :func:`~grstbx.subset.subset_path`
   * - GeoTIFF of :math:`R_{rs}`
     - Multi-band :math:`R_{rs}` for GIS software
     - one AOI, one date
     - path given by the caller, e.g. ``<product>_<site>_Rrs.tif``
   * - Datacube
     - Time series of a site
     - one site, one period
     - path given by the caller, e.g. ``<site>_<start>_<end>_box<size>m.nc`` in the tutorials
   * - Dashboard exports
     - Values of points, statistics of polygons, geometries
     - one session
     - ``points_values.csv``, ``polygons_statistics.csv``, ``points.geojson``, ``polygons.geojson``
