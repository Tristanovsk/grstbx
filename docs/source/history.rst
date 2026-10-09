History
=======

v2.3.0
   Multiscale interactive viewers: :class:`~grstbx.visual.ViewSpectral` and
   :class:`~grstbx.visual.ViewParam` read only the pixels visible on the screen, at the screen
   resolution, from the pyramids of the Zarr products (or from coarser levels computed on demand,
   :class:`~grstbx.visual.Multiscale`); only the visible window is reprojected to web Mercator.
   They open Zarr stores directly (path or list of paths). ``ViewSpectral`` adds a true-colour
   composite and the full-resolution spectra of the drawn points. Free basemaps: CARTO, OpenStreetMap,
   OpenTopoMap, Esri and the Stamen styles of Stadia Maps (:data:`~grstbx.visual.BASEMAPS`).
   Drawn geometries are returned in the right CRS when the map is not reprojected.

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
