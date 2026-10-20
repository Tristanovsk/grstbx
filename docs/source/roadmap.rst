Roadmap
=======

This page describes the next developments of grstbx, to work with GRS products stored in object storage
(S3 buckets) and indexed in STAC catalogues, as data lakes and cloud platforms do. The priorities and the
schedule may change: comments and contributions are welcome on the
`issue tracker <https://github.com/Tristanovsk/grstbx/issues>`__.

Current state
-------------

GRS already writes cloud-ready products: Zarr stores with overviews (pyramids), chunks packed into shards
and consolidated metadata, that can be written directly to object storage (see the
`processing chain of GRS <https://grs.readthedocs.io/en/latest/processing_chain.html>`__). On the grstbx
side (version 2.4):

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 35 15 50

   * - Capability
     - Status
     - Notes
   * - Lazy reading of Zarr products, chunk by chunk, at the zoom level of the screen
     - available
     - :func:`~grstbx.driver.open_zarr_image`, :class:`~grstbx.visual.Multiscale`, viewers and dashboard
   * - Opening a Zarr store from a URL (``s3://``, ``gs://``, ``https://``...)
     - partial
     - :func:`~grstbx.driver.open_zarr_image` and :func:`~grstbx.driver.zarr_levels` go through fsspec:
       checked with ``file://`` URLs, not yet tested on a bucket; credentials only from the environment
       (no ``storage_options`` parameter)
   * - Discovery of products (:func:`~grstbx.app.find_products`, :class:`~grstbx.datalake.SelectFiles`,
       glob patterns, file browser of the dashboard)
     - local only
     - local file system (``os``, ``glob``); products selected from their file names
   * - :func:`~grstbx.app.open_products` and ``grstbx-viewer``
     - local only
     - paths checked on the local file system
   * - NetCDF products
     - local only
     - read with the netCDF4 library
   * - Exports (:func:`~grstbx.subset.export_l2a`, :func:`~grstbx.subset.crop_and_export`)
     - local only
     - output paths checked on the local file system
   * - STAC catalogues
     - not available
     -

Object storage (S3)
-------------------

Goal: open, explore and subset the products stored in S3-compatible buckets (AWS, but also the object
storage of data centres and cloud providers, or MinIO) as easily as local files, reading only the chunks
needed.

- **Storage options**: a ``storage_options`` parameter (endpoint URL of non-AWS services, credentials
  profile, anonymous access) passed down to fsspec by :func:`~grstbx.driver.open_zarr_image`,
  :func:`~grstbx.driver.zarr_levels`, :class:`~grstbx.driver.L2grs`, :func:`~grstbx.subset.open_l2a`,
  :func:`~grstbx.app.open_products` and :meth:`Multiscale.from_zarr <grstbx.visual.Multiscale.from_zarr>`;
  credentials read from the standard sources (environment variables, configuration files of the
  machine), never stored in the code or sent to the browser.
- **Paths through fsspec**: replace the local ``os.path`` and ``glob`` calls by fsspec file systems, so that
  :func:`~grstbx.app.find_products`, :func:`~grstbx.app.catalogue` and the glob patterns work on bucket
  prefixes, e.g. ``s3://bucket/L2A/31TEJ/2025/`` with the tree ``<tile>/YYYY/MM/DD/<product>``.
- **Dashboard**: URLs of products and prefixes in the *Products* field, browser of buckets in the sidebar,
  credentials configured on the server. Served close to the data (same cloud region), the dashboard
  avoids transferring the products to the user.
- **Performance**: the consolidated metadata and the shards of the GRS products limit the number of
  requests; add the caching of fsspec (block cache) for the live spectra and time series, which read the
  same chunks again, and the parallel opening of long time series.
- **Exports to buckets**: subsets and datacubes written as Zarr stores to ``s3://`` paths
  (:func:`~grstbx.subset.export_l2a` uses the writers of GRS, which already write to object storage).
- **NetCDF products**: readable from buckets through h5netcdf and fsspec, but slowly (many small
  requests): converting them to Zarr will be recommended.
- **Tests**: automatic tests against a local S3 service (MinIO).

STAC catalogue
--------------

Goal: find the products with a `STAC <https://stacspec.org>`__ catalogue (SpatioTemporal Asset Catalog)
instead of file names, and open the results directly as time series.

- **Search**: query a STAC API with ``pystac-client`` by collection (GRS L2A, L2B), area of interest,
  period, cloud cover (``eo:cloud_cover``), tile (``grid:code``) and processing version
  (``processing:version``), as an alternative backend of :class:`~grstbx.datalake.SelectFiles` and
  :func:`~grstbx.app.select_products`.
- **From items to time series**: open the Zarr assets of the items (``href`` on a bucket, Zarr media
  type) with :func:`~grstbx.app.open_products`. The metadata of the items (acquisition time, footprint,
  ``proj:code``, ``proj:shape``, ``proj:transform``) give the dates, grids and CRS without reading the
  products, so that the selection of the tile and the lazy opening of long time series stay instant; the
  opening options can come from the ``xarray`` assets extension.
- **Generation of the catalogue**: create the STAC items and collections of GRS products with ``pystac``,
  from the attributes of the Zarr stores: eo (bands), projection, processing, sat and view (angles)
  extensions, links from the L2B products to the L2A products they are derived from; published as a
  static catalogue (JSON files next to the products) or loaded into a STAC API. Writing the item at the
  end of the processing would rather belong to GRS: to be coordinated with the GRS processor. GRSl2bgen
  (>= 1.1) already writes the Item of each L2B product, ``<name>/<name>.json`` (projection, processing,
  file, eo and grid extensions, ``derived_from`` link to the L2A), written last. grstbx reads these
  Items (:func:`~grstbx.app.item_info`): dates, tiles, versions and cloud cover come from them, and the
  time series can be filtered by cloud cover; the search in a STAC API remains to do.
- **Dashboard**: a *STAC search* card (catalogue URL, collection, area of interest drawn on the map,
  period, maximum cloud cover) listing the matching items, opened as a time series like the folders
  today. *Done* (:func:`~grstbx.app.search_items`, ``grstbx-viewer --stac``): local Items or STAC API
  (Item Search ``POST /search`` with paging, plain HTTP, no ``pystac-client``). To do: opening remote
  data assets (``https``/``s3`` Zarr stores) is not tested yet, the search of
  :class:`~grstbx.datalake.SelectFiles`, and the ``processing:version`` criterion.

Milestones
----------

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 8 42 50

   * - Step
     - Development
     - Content
   * - 1
     - Remote Zarr products
     - ``storage_options``, fsspec paths in the Python API, tests on MinIO, optional dependencies in a
       ``cloud`` extra (s3fs)
   * - 2
     - Discovery and dashboard on buckets
     - :func:`~grstbx.app.find_products` on bucket prefixes, URLs and bucket browser in ``grstbx-viewer``,
       caching of the live spectra and time series
   * - 3
     - STAC search
     - search and opening of the results as time series in Python and in the dashboard
       (``pystac-client``)
   * - 4
     - STAC items of GRS products
     - generation of the items and collections (``pystac``), static catalogue or STAC API, with GRS
   * - 5
     - Exports to buckets
     - subsets and datacubes written as Zarr stores to object storage
