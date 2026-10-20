History
=======

Unreleased
   - ``grstbx-viewer``: time series from folders, e.g. the tree ``<tile>/YYYY/MM/DD/<product>`` of the GRS
     batches (:func:`~grstbx.app.find_products`), with the GRS >= 3 product folders ``<name>/<name>.zarr``;
     *Time series* card (and ``--tile``, ``--start``, ``--end``, ``--all-processings``) to select the tile
     (required when the folders hold several UTM zones), the period and the last processing of each
     acquisition (:func:`~grstbx.app.select_products`). The products of a series are opened when they are
     displayed, so that hundreds of products open in a second; for more than 30 dates the live time series
     of L2B is off at opening. The level of the products is checked on their content when their names
     disagree.
   - ``grstbx-viewer``: the invalid pixels (``mask != 0``) of the L2A products are hidden too, by default
     (option of the *Options* card, ``--show-invalid-l2a``; ``--show-invalid-l2b`` for the L2B products;
     :func:`~grstbx.app.open_products` ``mask_l2a=``).
   - Viewers: the colorbar is hidden when the RGB composite is displayed (it showed a meaningless -1 to 1
     scale, or the range of the band displayed before) and shown again for the single bands and
     parameters (:func:`~grstbx.visual.colorbar_visibility_hook`).
   - :class:`~grstbx.visual.Multiscale`: the windows of the map are read with a margin of 25 % and kept, so
     that panning and zooming inside it read no data (about 3 times faster); the caches (windows, coarser
     levels computed for the images without pyramid, blocks of the spectra and time series) are bounded in
     memory (``cache_size``, 2 GB by default, least recently used data dropped first), they were unbounded
     or bounded in number only; ``grstbx-viewer --cache-size`` (GB per session).
   - :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`: ``live=`` option (initial
     state of the live spectrum / time series); widgets on four rows, so that the *Live*, *Reverse* and
     *Log scale* checkboxes are no longer cut at the right of the map.

   - :meth:`L2grs.get_l2b_datacube <grstbx.driver.L2grs.get_l2b_datacube>`: ``variables=`` option to read
     only the variables needed from the large L2B products; the scalar coordinates of the ancillary data
     (e.g. ``forecast_period``, stored as a float or a timedelta depending on the products) are no longer
     concatenated along time, which failed for such mixed series.
   - Documentation: case study of the L2B time series of the Lac de Sainte-Croix (tile 31TGJ),
     ``notebook/case_study/sainte_croix/grstbx_l2b_sainte_croix.ipynb``, replacing the Lac de Saint-Cassien
     notebook ``notebook/validation/grstbx_l2b_matchup.ipynb``.

v2.4.0
   First release published on PyPI (``pip install grstbx``).

   - Web dashboard ``grstbx-viewer`` (:mod:`grstbx.app`): opens GRS L2A and L2B products (Zarr stores,
     NetCDF products, hGRS L2A files), one image, a time series of products (paths or glob patterns,
     possibly on different grids) or a NetCDF datacube, chosen in a file browser of the server; the L2A
     products are displayed with :class:`~grstbx.visual.ViewSpectral`, the L2B products with
     :class:`~grstbx.visual.ViewParam` (invalid pixels hidden); exports of the full-resolution values of
     the drawn points and of the statistics of the drawn polygons for all the dates (CSV) and of the
     geometries (GeoJSON); it can be served on a network (``--address``).
   - :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`: contrast and brightness
     sliders (RGB composite and colour range of the single bands or parameters); ViewSpectral: bands of
     the RGB composite (presets: true colour, false colour, red edge, SWIR, or any band for each channel)
     and gamma. The widgets are shown in the viewer, or placed elsewhere with ``display_controls()`` and
     ``rgb_controls()`` (``visu(controls=False)``), as in the sidebar of ``grstbx-viewer``. ``title=``
     option to name the viewer after the sensor (e.g. ``'## PRISMA L2A'``; default ``'## S2 L2A'`` and
     ``'## S2 L2B'``).
   - hGRS products (hyperspectral GRS, EnMAP and PRISMA): :func:`~grstbx.driver.open_hgrs_l2a` opens an
     hGRS L2A NetCDF file as a product (full-resolution ``Rrs`` and ``brdfg_full``) and its ancillary
     data (aerosol, water vapor and sunglint on the coarse grid ``xc``, ``yc``), like the GRS products, so
     that it can be cropped with :func:`~grstbx.subset.crop_l2a` and displayed with the viewers; the
     SNAP-compliant hGRS products (one ``Rrs_<wl>`` variable per band) are reshaped into ``Rrs``.
     :func:`~grstbx.subset.open_l2a` opens GRS and hGRS products (:func:`~grstbx.driver.is_hgrs`).
   - Fix: :meth:`SpatioTemp.clip_raster <grstbx.utils.SpatioTemp.clip_raster>` clipped a square of side
     1.41 ``extent_m`` instead of ``extent_m``; the clipped areas of former runs are 2 times larger
     (:meth:`SpatioTemp.wktbox <grstbx.utils.SpatioTemp.wktbox>` has had the same correction since
     v2.2.0: it returned a box twice the requested ``width`` and ``height`` before).
   - Fix: :func:`~grstbx.utils.utm_projection` returns the plate carrée projection for
     longitude-latitude grids (e.g. EPSG 4326 of the hGRS PRISMA products), which
     :func:`~grstbx.utils.plot_rgb` drew as web Mercator coordinates.
   - Fix: the drop-down menus of :class:`~grstbx.visual.ViewSpectral` and :class:`~grstbx.visual.ViewParam`
     (colormap picker) were clipped by the widget box: they are now drawn above the map.
   - Docs: tutorial pages on the web dashboard and on the hGRS products, with the PRISMA L1-to-L2A
     notebook (L1 and L2C inputs interpolated onto the L2A grid, composites, retrieved parameters,
     spectra, transect) and a link to the EnMAP tutorial of hGRS; harmonized titles, sections and
     captions of the tutorial notebooks.
   - Packaging: source distribution without tests, notebooks, docs and illustrations; GitHub Actions
     workflow publishing to PyPI on ``v*`` tags (Trusted Publishing).

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
