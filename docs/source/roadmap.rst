Roadmap
=======

This page describes the next developments of grstbx: GRS products stored in object storage (S3 buckets) and
indexed in STAC catalogues, as data lakes and cloud platforms do, the validation of the products with in situ
networks, and the processing on HPC clusters. The priorities and the schedule may change: comments and
contributions are welcome on the `issue tracker <https://github.com/Tristanovsk/grstbx/issues>`__.

*Last update: November 2026 (grstbx 2.8.0.dev0).*

Current state
-------------

GRS already writes cloud-ready products: Zarr stores with overviews (pyramids), chunks packed into shards
and consolidated metadata, that can be written directly to object storage (see the
`processing chain of GRS <https://grs.readthedocs.io/en/latest/processing_chain.html>`__). On the grstbx
side (version 2.8, development):

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
   * - Opening a product from a URL (``s3://``, ``https://``...)
     - available
     - :func:`~grstbx.app.open_products` and the viewers open Zarr stores and NetCDF files of buckets and
       web servers through fsspec, tested on a local S3 service (``rclone serve s3``) with the credentials and
       endpoint of an AWS profile; credentials only from the environment (no ``storage_options`` parameter
       in :func:`~grstbx.driver.open_zarr_image`, :func:`~grstbx.driver.zarr_levels`,
       :class:`~grstbx.driver.L2grs`)
   * - Discovery of products (:func:`~grstbx.app.find_products`, :class:`~grstbx.datalake.SelectFiles`,
       glob patterns, file browser of the dashboard)
     - local only
     - local file system (``os``, ``glob``); products selected from their file names
   * - ``grstbx-viewer``
     - partial
     - products opened from URLs or from the results of a STAC search; the file browser explores local
       folders only
   * - NetCDF products
     - available
     - netCDF4 locally, h5netcdf through fsspec in buckets (slow: many small requests)
   * - Exports (:func:`~grstbx.subset.export_l2a`, :func:`~grstbx.subset.crop_and_export`)
     - local only
     - output paths checked on the local file system
   * - Batch processing (:mod:`grstbx.batch`)
     - available
     - inputs and outputs in local folders or buckets (fsspec), staged in a local work folder; L2A STAC Items
       (:func:`~grstbx.batch.write_l2a_item`), images without water marked and skipped
   * - Download of Sentinel-2 images (:mod:`grstbx.download`)
     - available
     - search in the catalogue of the Copernicus Data Space Ecosystem and download from its S3 interface
       into the tree of the batches
   * - Local S3 bucket (``grstbx-s3-bucket``)
     - available
     - folders served as a bucket (``rclone serve s3``), one prefix per sensor and level, with its AWS profile
   * - STAC catalogues
     - available
     - static catalogue (:func:`~grstbx.stac.build_catalog`, :func:`~grstbx.stac.bucket_items`), search in
       Python and in the dashboard (:func:`~grstbx.app.search_items`,
       :meth:`~grstbx.datalake.SelectFiles.search_stac`), see below
   * - Validation with in situ data (:mod:`grstbx.validation`)
     - available
     - AERONET-OC and HYPERNETS matchups on Sentinel-2 subsets, dashboard, see below
   * - HPC (CNES TREX)
     - available
     - batches from the CNES datalake and exports, as SLURM job arrays (notebooks, see :doc:`hpc/cnes_trex`)

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
- **Batches on buckets**: *done* (:mod:`grstbx.batch`, see :doc:`tutorials/batch`): Level-1 images, L2A
  and L2B products listed, checked and written in buckets (``s3://`` roots of :class:`~grstbx.batch.Batch`,
  ``storage_options``), each image staged in a local work folder for GRS and GRSl2bgen, the product folders
  uploaded with their STAC Item last. To do: GRS and GRSl2bgen reading and writing the buckets directly,
  without the local copies.
- **Exports to buckets**: subsets and datacubes written as Zarr stores to ``s3://`` paths
  (:func:`~grstbx.subset.export_l2a` uses the writers of GRS, which already write to object storage).
- **NetCDF products**: readable from buckets through h5netcdf and fsspec, but slowly (many small
  requests): converting them to Zarr will be recommended.
- **Tests**: *done*: automatic tests against a local S3 service (``rclone serve s3``, :mod:`grstbx.bucket`)
  with the credentials and endpoint of an AWS profile. To do: a real bucket (CNES datalake, cloud provider).

STAC catalogue
--------------

Goal: find the products with a `STAC <https://stacspec.org>`__ catalogue (SpatioTemporal Asset Catalog)
instead of file names, and open the results directly as time series.

- **Search**: *done* without ``pystac-client`` (plain HTTP, Item Search ``POST /search`` with paging, or the
  Items of a static catalogue): by collection (GRS L2A, L2B), area of interest, period, cloud cover
  (``eo:cloud_cover``), tile (``grid:code``) and GRS version (``processing:software``), in
  :func:`~grstbx.app.search_items` and :meth:`~grstbx.datalake.SelectFiles.search_stac`.
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
  (Item Search ``POST /search`` with paging, plain HTTP, no ``pystac-client``), remote Items and Zarr
  assets over ``https`` (tested with GRS L2A and L2B tiles served over HTTP), and
  :meth:`~grstbx.datalake.SelectFiles.search_stac` (with the GRS version criterion, from
  ``processing:software``). To do: ``s3://`` assets on a real bucket (credentials).
- **Static catalogue in a bucket**: *done* (:func:`~grstbx.stac.build_catalog`, see
  :doc:`tutorials/s3_stac`): the collections ``grs-l2a`` and ``grs-l2b`` with ``s3://`` assets, searched by
  :func:`~grstbx.app.search_items` and tested on a local S3 service (``rclone serve s3``), with credentials
  and endpoint from an AWS profile. The catalogue can also be built from the STAC Items read in the bucket
  (:func:`~grstbx.stac.bucket_items`) and written to it, after batches writing to the bucket: L2B Items of
  GRSl2bgen, L2A Items of :func:`~grstbx.batch.write_l2a_item` (``l2a_stac=True``), with the PNG quicklooks
  of GRS as assets and the L2A RGB as thumbnail. To do: ``storage_options`` for the search and the viewers, a
  real bucket.

Validation with in situ data
----------------------------

Goal: validate each GRS version with the in situ networks, on Sentinel-2 subsets small enough to process the
whole archive of the stations in a few hours, with the same matchup table, quality criteria, statistics and
dashboard for all the networks.

- **AERONET-OC**: *done* (:mod:`grstbx.validation`, ``grstbx-validation``, see :doc:`tutorials/aeronet_oc_validation`):
  subsets of 5 km around the stations in the bucket ``aeronet-oc`` with their STAC catalogue, GRS on the
  subsets (one process per image), ring of water pixels around the platform, nearest Level 2.0 measurement,
  three in situ normalisations, quality criteria kept as columns, statistics by band, station and aerosol
  model, dashboard (``grstbx-validation dashboard``) to compare GRS versions.
- **HYPERNETS**: *done* (:func:`~grstbx.validation.read_hypernets`, see :doc:`tutorials/validation`):
  QC-approved spectra of the in situ validation database convolved with the spectral response functions of
  the Sentinel-2 bands, normalisation to nadir viewing, the same chain on subsets of 5 km around BEFR and
  MAFR. To do: the other WATERHYPERNET sites (VEIT, GAIT, LPAR, M1BE...) and the 2023-2024 archive, a STAC
  catalogue of the HYPERNETS subsets as for AERONET-OC, the SRF of each platform (S2A, S2B, S2C).
- **L2B products**: matchups of the GRSl2bgen parameters (chlorophyll-a, suspended matter, turbidity) with the
  in situ samples of the validation database (REPHY, SOMLIT, GLORIA...), in the same table format.
- **Other sensors**: Landsat 8-9 and the hyperspectral sensors of hGRS (EnMAP, PRISMA), with their spectral
  response functions.

HPC
---

Goal: process and export the archive of the data lakes of the computing centres with the same notebooks as
on a workstation, run as SLURM job arrays.

- **CNES TREX**: *done* (see :doc:`hpc/cnes_trex`): L1C → L2A (GRS 2.2 or 3.x) → L2B (GRSl2bgen) from the
  CNES datalake (temporary credentials of a read-only role renewed during the batch, restoration of the
  images archived on tier 3), products written to ``/work/datalake``, SLURM job array on a tile × month grid
  with a time budget, export of an area to Google Drive.
- **To do**: the batch functions of the notebooks (datalake access, tier 3, time budget) moved to
  :mod:`grstbx.batch`; the validation chain on TREX for the whole archive of the stations; the products
  written to the datalake as Zarr stores with their STAC catalogue.

Milestones
----------

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 8 30 14 48

   * - Step
     - Development
     - Status
     - Content
   * - 1
     - Remote Zarr products
     - partial
     - done: products opened from URLs, tests on a local S3 service, ``remote`` extra (fsspec, s3fs,
       h5netcdf); to do: ``storage_options`` in the Python API
   * - 2
     - Discovery and dashboard on buckets
     - partial
     - done: products of a STAC search opened in ``grstbx-viewer``, time series read in parallel; to do:
       :func:`~grstbx.app.find_products` on bucket prefixes, bucket browser, caching of the live spectra
   * - 3
     - STAC search
     - done
     - search and opening of the results as time series in Python and in the dashboard (plain HTTP, static
       catalogues and STAC APIs)
   * - 4
     - STAC items of GRS products
     - done
     - L2A Items (grstbx) and L2B Items (GRSl2bgen), static catalogue in a bucket; to do with GRS: the L2A
       Item written by the processor
   * - 5
     - Exports to buckets
     - to do
     - subsets and datacubes written as Zarr stores to object storage
   * - 6
     - Validation with in situ networks
     - partial
     - done: AERONET-OC and HYPERNETS (BEFR, MAFR) on Sentinel-2 subsets, dashboard; to do: all the
       HYPERNETS sites, L2B parameters, other sensors
   * - 7
     - HPC
     - partial
     - done: CNES TREX notebooks (batch, export); to do: batch functions in :mod:`grstbx.batch`, validation
       of the whole archive on TREX
