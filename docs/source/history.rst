History
=======

Unreleased
   Web dashboard ``grstbx-viewer`` (:mod:`grstbx.app`): opens GRS L2A and L2B products (Zarr stores,
   NetCDF products, hGRS L2A files), one image, a time series of products (paths or glob patterns,
   possibly on different grids) or a NetCDF datacube, chosen in a file browser of the server; the L2A
   products are displayed with :class:`~grstbx.visual.ViewSpectral`, the L2B products with
   :class:`~grstbx.visual.ViewParam` (invalid pixels hidden); exports of the full-resolution values of the
   drawn points and of the statistics of the drawn polygons for all the dates (CSV) and of the geometries
   (GeoJSON); it can be served on a network (``--address``). Docs: tutorial page on the dashboard.
   Fix: the drop-down menus of :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`
   (colormap picker) were clipped by the widget box: they are now drawn above the map.
   :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`: contrast and brightness
   sliders (RGB composite and colour range of the single bands or parameters); ViewSpectral: bands of the
   RGB composite (presets: true colour, false colour, red edge, SWIR, or any band for each channel) and
   gamma. The widgets are shown in the viewer, or placed elsewhere with ``display_controls()`` and
   ``rgb_controls()`` (``visu(controls=False)``), as in the sidebar of ``grstbx-viewer``.
   hGRS products (hyperspectral GRS, EnMAP and PRISMA): :func:`~grstbx.driver.open_hgrs_l2a` opens an
   hGRS L2A NetCDF file as a product (full-resolution ``Rrs`` and ``brdfg_full``) and its ancillary data
   (aerosol, water vapor and sunglint on the coarse grid ``xc``, ``yc``), like the GRS products, so that
   it can be cropped with :func:`~grstbx.subset.crop_l2a` and displayed with the viewers; the
   SNAP-compliant hGRS products (one ``Rrs_<wl>`` variable per band) are reshaped into ``Rrs``.
   :func:`~grstbx.subset.open_l2a` opens GRS and hGRS products (:func:`~grstbx.driver.is_hgrs`).
   Fix: :func:`~grstbx.utils.utm_projection` returns the plate carrée projection for longitude-latitude
   grids (e.g. EPSG 4326 of the hGRS PRISMA products), which :func:`~grstbx.utils.plot_rgb` drew as web
   Mercator coordinates.
   Fix: :meth:`SpatioTemp.clip_raster <grstbx.utils.SpatioTemp.clip_raster>` clipped a square of side
   1.41 ``extent_m`` instead of ``extent_m``; the clipped areas of former runs are 2 times larger
   (:meth:`SpatioTemp.wktbox <grstbx.utils.SpatioTemp.wktbox>` has had the same correction since
   v2.2.0: it returned a box twice the requested ``width`` and ``height`` before).
   :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`: ``title=`` option to
   name the viewer after the sensor (e.g. ``'## PRISMA L2A'``; default ``'## S2 L2A'`` and
   ``'## S2 L2B'``).
   Docs: tutorial page on the hGRS products, with the PRISMA L1-to-L2A notebook (L1 and L2C inputs
   interpolated onto the L2A grid, composites, retrieved parameters, spectra, transect) and a link to
   the EnMAP tutorial of hGRS.

v2.3.2
   Matchups between satellite and in situ data (:mod:`grstbx.matchup`):
   :func:`~grstbx.matchup.get_matchups` pairs each in situ sample with the closest image within a
   maximum time lag, :func:`~grstbx.matchup.matchup_metrics` computes the agreement statistics
   (regression, rmse, bias, mape, median symmetric accuracy and signed symmetric percentage bias) and
   :func:`~grstbx.matchup.plot_matchup` draws the scatter plot with these statistics. They replace the
   functions defined in the L2B validation tutorial.

v2.3.1
   :class:`~grstbx.visual.ViewSpectral`: live full-resolution spectrum of the pixel under the mouse
   (*Live spectrum* checkbox), plotted with the spectra of the drawn points; the pixels are read by
   blocks aligned on the Zarr chunks, with all the bands, and the last blocks are kept in memory
   (:meth:`~grstbx.visual.Multiscale.pixel`). :class:`~grstbx.visual.ViewParam`: same colormaps as
   ``ViewSpectral`` (:data:`~grstbx.visual.RRS_COLORMAPS`, previews, reverse toggle) and ``cmap=``
   option; the former list ``COLORMAPS`` is removed. ``ViewParam``: time series of the parameter at
   the drawn points and live time series of the pixel under the mouse
   (:meth:`~grstbx.visual.Multiscale.timeseries`). Both viewers: ``aoi=`` and ``poi=`` options to
   draw initial polygons and points (GeoDataFrame, shapely geometry, WKT or (lon, lat)); the points
   keep the same colour as their spectra and time series; plots are placed beneath the map.
   Default basemap: OpenStreetMap, which needs no API key (the Stadia Maps tiles return error 401
   outside ``localhost``, e.g. on Read the Docs). New tutorial: GRS L2B datacube (Berre lagoon).

v2.3.0
   Multiscale interactive viewers: :class:`~grstbx.visual.ViewSpectral` and
   :class:`~grstbx.visual.ViewParam` read only the pixels visible on the screen, at the screen
   resolution, from the pyramids of the Zarr products (or from coarser levels computed on demand,
   :class:`~grstbx.visual.Multiscale`); only the visible window is reprojected to web Mercator.
   They open Zarr stores directly (path or list of paths). ``ViewSpectral`` adds a true-colour
   composite and the full-resolution spectra of the drawn points. Free basemaps: CARTO, OpenStreetMap,
   OpenTopoMap, Esri and the Stamen styles of Stadia Maps (:data:`~grstbx.visual.BASEMAPS`).
   Drawn geometries are returned in the right CRS when the map is not reprojected.
   ``basemap=`` sets the initial basemap. ``ViewSpectral`` colormaps: sequential, perceptually
   uniform maps named after the water feature they enhance (:data:`~grstbx.visual.RRS_COLORMAPS`,
   colorcet and cmocean), with previews, a reverse toggle and ``cmap=``; fix of the crash when
   switching between the RGB composite and a single band.
   Faster datacubes (:class:`~grstbx.driver.L2grs`): images opened with the chunks of the files,
   flag statistics and subsets of all the images computed in a single dask call (one task per
   image), ancillary files skipped, ``scheduler`` option (single thread by default for netCDF,
   whose HDF5 reads are serialized); about 4x faster to build and 5x faster to read.

v2.2.0
   Crop L2A products to an area of interest and export the subsets as GRS products (NetCDF or Zarr)
   or GeoTIFF (:mod:`grstbx.subset`); the ancillary group of Zarr products is opened with the image.
   Map helpers :func:`~grstbx.utils.utm_projection` (fixes the UTM zone / hemisphere detection) and
   :func:`~grstbx.utils.plot_rgb` (true-colour composite, decimated for large images, AOI outline).
   Packaging: dependencies with lower bounds, unused ones removed (``docopt``, ``pandas-bokeh``),
   ``netCDF4`` added, Jupyter moved to the ``notebook`` extra; ``environment.yml`` and
   ``requirements.txt``.

v2.1.1
   Open Zarr v3 multiscale (pyramid) images, groups ``'0'``, ``'1'``, ``'2'``, ...

v2.1.0
   Code optimization and packaging with ``pyproject.toml``.

v2.0.4
   Code optimization and documentation; lazy loading of the :mod:`~grstbx.visual` module.

v2.0.3 (2026-04-08)
   Tool option for masking / bitmask flagging.

v2.0.2
   Small changes for the visualization devices.

v2.0.1
   Fix for GDAL projection for accepted dtype, fix for dashboard visualization.

v2.0.0
   Transition to GRS v2.

v1.0.5
   Revisit datacube and raster objects for multi-tile access.

v1.0.4
   Tools for L2B handling.

v1.0.3
   Fix for reprojection in any system, especially useful for reprojection in pseudo-Mercator
   (EPSG:3857) for visualization with web mapping tools (e.g. OpenStreetMap, Google Maps).
