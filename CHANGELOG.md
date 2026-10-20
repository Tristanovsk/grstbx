# Changelog

All notable changes of grstbx. The format follows [Keep a Changelog](https://keepachangelog.com) and
the versions follow [Semantic Versioning](https://semver.org); each version is published on
[PyPI](https://pypi.org/project/grstbx/) from the git tag `vX.Y.Z`.

## [Unreleased]

### Added

- L2B product folders of GRSl2bgen (`--product_dir`): `<name>/` holding `<name>.nc` (or `<name>.zarr`),
  its log `<name>.log` and its STAC Item `<name>.json`, in the tree `<tile>/YYYY/MM/DD/<name>/`. They
  are found by `find_products` and the file browser, and opened from their NetCDF file (`grstbx-viewer
  /data/L2B --tile 31TFJ`), instead of being taken for GRS 2 L2A folders (which hold `<name>_anc.nc`).
- STAC Items of the products: a product with an Item (`<name>/<name>.json`, or `<name>.json` next to the
  product) is described from it rather than from its name (`product_info`, `item_info`): acquisition
  time (`datetime`), tile (`grid:code`), level (`processing:level`), GRS version (`processing:software`),
  and the new columns `cloud_cover` (`eo:cloud_cover`), `crs` (`proj:code`) and `item`. Items can be
  given directly, e.g. a glob pattern `.../*/*/*/*/*.json` (`is_stac_item`: the data asset, a relative
  `href`, must exist). Plain JSON reading, no new dependency.
- Maximum cloud cover of the products with a STAC Item: `select_products(max_cloud_cover=)`,
  `open_products(max_cloud_cover=)`, *Time series* card of the dashboard, `grstbx-viewer
  --max-cloud-cover`; the products without Item are kept.
- *STAC search* card of the dashboard (`grstbx-viewer --stac <catalogue>`): catalogue (STAC API URL, or
  folder / glob pattern of local Items), collections, area (everywhere, last polygon drawn on the map,
  or bounding box), period and maximum cloud cover; lists the matching products and opens them as a time
  series. `search_items` does the search: STAC API Item Search (`POST /search`, `next` links followed,
  cloud cover through the query extension), or local Items filtered by bbox, period, tile and cloud
  cover; plain HTTP (`urllib`), no new dependency. Remote data assets of a STAC API go to the *Products*
  field as their Item URLs (`expand_paths` and `catalogue` accept URLs).
- Remote products: Zarr stores over `https` (and `s3`) and remote STAC Items (`https://.../item.json`
  or the `.../items/<id>` of a STAC API), fetched and described from the Item, their relative data
  asset resolved from the Item URL. Tested with GRS L2A (sharded, read with byte ranges) and L2B
  (pyramid) full tiles served over HTTP: a tile opens in about 2 s (7 to 9 requests, about 1 MB) and a
  window at full resolution is read in 0.3 s, with the same values as the local files. Optional
  dependencies `grstbx[remote]` (fsspec, aiohttp, s3fs).

## [2.5.0] - 2026-10-20

Time series of folders in the dashboard, faster viewers with bounded caches, tests and continuous
integration.

### Added

- `grstbx-viewer`: time series from folders, e.g. the tree `<tile>/YYYY/MM/DD/<product>` of the GRS
  batches (`find_products`), with the GRS >= 3 product folders `<name>/<name>.zarr`. The *Time series*
  card (and `--tile`, `--start`, `--end`, `--all-processings`) selects the tile (required when the
  folders hold several UTM zones), the period and the last processing of each acquisition
  (`select_products`). The products of a series are opened when they are displayed, so that hundreds of
  products open in a second; for more than 30 dates the live time series of L2B is off at opening.
- `grstbx-viewer`: the invalid pixels (`mask != 0`) of the L2A products are hidden too, by default
  (option of the *Options* card, `--show-invalid-l2a`; `--show-invalid-l2b` for the L2B products;
  `open_products(mask_l2a=)`).
- `ViewSpectral` and `ViewParam`: `live=` option (initial state of the live spectrum / time series).
- `L2grs.get_l2b_datacube`: `variables=` option to read only the variables needed from the large L2B
  products.
- Tests on synthetic data (`tests/`, `pip install "grstbx[test]"`, `pytest`) and continuous integration
  (`.github/workflows/ci.yml`): build, tests and command line of the installed wheel on Python 3.11 to
  3.13, and build of the documentation, for each push and pull request.
- Documentation: case study of the L2B time series of the Lac de Sainte-Croix (tile 31TGJ), replacing
  the Lac de Saint-Cassien notebook; tutorial pages on the template of the GRS examples, processing
  chain description, utilities tutorial illustrated with figures, roadmap for S3 object storage and STAC
  catalogues; this changelog.

### Changed

- `Multiscale`: the windows of the map are read with a margin of 25 % and kept, so that panning and
  zooming inside it read no data (about 3 times faster). The caches (windows, coarser levels computed
  for the images without pyramid, blocks of the spectra and time series) are bounded in memory
  (`cache_size`, 2 GB by default, least recently used data dropped first); they were unbounded or
  bounded in number only. `grstbx-viewer --cache-size` (GB per session).
- Viewers: widgets on four rows.
- Documentation: the notebooks of the tutorials are kept (and tracked) in `docs/source/tutorials/`, they
  are no longer copied from `notebook/` at each build. Read the Docs builds the HTML and PDF
  documentation only (no EPUB).

### Fixed

- Viewers: the colorbar is hidden when the RGB composite is displayed (it showed a meaningless -1 to 1
  scale, or the range of the band displayed before) and shown again for the single bands and parameters
  (`colorbar_visibility_hook`).
- Viewers: the *Live*, *Reverse* and *Log scale* checkboxes were cut at the right of the map.
- `L2grs.get_l2b_datacube`: the scalar coordinates of the ancillary data (e.g. `forecast_period` of
  CAMS, stored as a float or a timedelta depending on the products) are no longer concatenated along
  time, which failed for such mixed series.
- `grstbx-viewer`: the level of the products (L2A, L2B) is checked on their content when their names
  disagree.

## [2.4.0] - 2026-10-16

First release published on PyPI (`pip install grstbx`).

### Added

- Web dashboard `grstbx-viewer` (`grstbx.app`): opens GRS L2A and L2B products (Zarr stores, NetCDF
  products, hGRS L2A files), one image, a time series of products (paths or glob patterns, possibly on
  different grids) or a NetCDF datacube, chosen in a file browser of the server; the L2A products are
  displayed with `ViewSpectral`, the L2B products with `ViewParam` (invalid pixels hidden); exports of
  the full-resolution values of the drawn points and of the statistics of the drawn polygons for all the
  dates (CSV) and of the geometries (GeoJSON); it can be served on a network (`--address`).
- `ViewSpectral` and `ViewParam`: contrast and brightness sliders (RGB composite and colour range of the
  single bands or parameters); `ViewSpectral`: bands of the RGB composite (presets: true colour, false
  colour, red edge, SWIR, or any band for each channel) and gamma. The widgets are shown in the viewer,
  or placed elsewhere with `display_controls()` and `rgb_controls()` (`visu(controls=False)`), as in the
  sidebar of `grstbx-viewer`. `title=` option to name the viewer after the sensor (e.g.
  `'## PRISMA L2A'`; default `'## S2 L2A'` and `'## S2 L2B'`).
- hGRS products (hyperspectral GRS, EnMAP and PRISMA): `open_hgrs_l2a` opens an hGRS L2A NetCDF file as
  a product (full-resolution `Rrs` and `brdfg_full`) and its ancillary data (aerosol, water vapor and
  sunglint on the coarse grid `xc`, `yc`), like the GRS products, so that it can be cropped with
  `crop_l2a` and displayed with the viewers; the SNAP-compliant hGRS products (one `Rrs_<wl>` variable
  per band) are reshaped into `Rrs`. `open_l2a` opens GRS and hGRS products (`is_hgrs`).
- Documentation: tutorial pages on the web dashboard and on the hGRS products, with the PRISMA
  L1-to-L2A notebook (L1 and L2C inputs interpolated onto the L2A grid, composites, retrieved
  parameters, spectra, transect) and a link to the EnMAP tutorial of hGRS.
- Packaging: GitHub Actions workflow publishing to PyPI on `v*` tags (Trusted Publishing) and creating
  the GitHub release.

### Changed

- Documentation: harmonized titles, sections and captions of the tutorial notebooks.
- Packaging: source distribution without tests, notebooks, docs and illustrations.

### Fixed

- `SpatioTemp.clip_raster` clipped a square of side 1.41 `extent_m` instead of `extent_m`; the clipped
  areas of former runs are 2 times larger (`SpatioTemp.wktbox` has had the same correction since
  v2.2.0: it returned a box twice the requested `width` and `height` before).
- `utm_projection` returns the plate carrée projection for longitude-latitude grids (e.g. EPSG 4326 of
  the hGRS PRISMA products), which `plot_rgb` drew as web Mercator coordinates.
- The drop-down menus of `ViewSpectral` and `ViewParam` (colormap picker) were clipped by the widget
  box: they are now drawn above the map.

## 2.3.2

Matchups between satellite and in situ data (`grstbx.matchup`): `get_matchups` pairs each in situ sample
with the closest image within a maximum time lag, `matchup_metrics` computes the agreement statistics
(regression, rmse, bias, mape, median symmetric accuracy and signed symmetric percentage bias) and
`plot_matchup` draws the scatter plot with these statistics. They replace the functions defined in the
L2B validation tutorial.

## 2.3.1

`ViewSpectral`: live full-resolution spectrum of the pixel under the mouse (*Live spectrum* checkbox),
plotted with the spectra of the drawn points; the pixels are read by blocks aligned on the Zarr chunks,
with all the bands, and the last blocks are kept in memory (`Multiscale.pixel`). `ViewParam`: same
colormaps as `ViewSpectral` (`RRS_COLORMAPS`, previews, reverse toggle) and `cmap=` option; the former
list `COLORMAPS` is removed. `ViewParam`: time series of the parameter at the drawn points and live time
series of the pixel under the mouse (`Multiscale.timeseries`). Both viewers: `aoi=` and `poi=` options to
draw initial polygons and points (GeoDataFrame, shapely geometry, WKT or (lon, lat)); the points keep the
same colour as their spectra and time series; plots are placed beneath the map. Default basemap:
OpenStreetMap, which needs no API key (the Stadia Maps tiles return error 401 outside `localhost`, e.g.
on Read the Docs). New tutorial: GRS L2B datacube (Berre lagoon).

## [2.3.0] - 2026-10-09

Multiscale interactive viewers: `ViewSpectral` and `ViewParam` read only the pixels visible on the
screen, at the screen resolution, from the pyramids of the Zarr products (or from coarser levels computed
on demand, `Multiscale`); only the visible window is reprojected to web Mercator. They open Zarr stores
directly (path or list of paths). `ViewSpectral` adds a true-colour composite and the full-resolution
spectra of the drawn points. Free basemaps: CARTO, OpenStreetMap, OpenTopoMap, Esri and the Stamen styles
of Stadia Maps (`BASEMAPS`). Drawn geometries are returned in the right CRS when the map is not
reprojected. `basemap=` sets the initial basemap. `ViewSpectral` colormaps: sequential, perceptually
uniform maps named after the water feature they enhance (`RRS_COLORMAPS`, colorcet and cmocean), with
previews, a reverse toggle and `cmap=`; fix of the crash when switching between the RGB composite and a
single band. Faster datacubes (`L2grs`): images opened with the chunks of the files, flag statistics and
subsets of all the images computed in a single dask call (one task per image), ancillary files skipped,
`scheduler` option (single thread by default for netCDF, whose HDF5 reads are serialized); about 4x
faster to build and 5x faster to read.

## [2.2.0] - 2026-10-09

Crop L2A products to an area of interest and export the subsets as GRS products (NetCDF or Zarr) or
GeoTIFF (`grstbx.subset`); the ancillary group of Zarr products is opened with the image. Map helpers
`utm_projection` (fixes the UTM zone / hemisphere detection) and `plot_rgb` (true-colour composite,
decimated for large images, AOI outline). Packaging: dependencies with lower bounds, unused ones removed
(`docopt`, `pandas-bokeh`), `netCDF4` added, Jupyter moved to the `notebook` extra; `environment.yml` and
`requirements.txt`.

## [2.1.1] - 2026-10-08

Open Zarr v3 multiscale (pyramid) images, groups `'0'`, `'1'`, `'2'`, ...

## 2.1.0

Code optimization and packaging with `pyproject.toml`.

## 2.0.4

Code optimization and documentation; lazy loading of the `visual` module.

## [2.0.3] - 2026-04-08

Tool option for masking / bitmask flagging.

## 2.0.2

Small changes for the visualization devices.

## 2.0.1

Fix for GDAL projection for accepted dtype, fix for dashboard visualization.

## 2.0.0

Transition to GRS v2.

## 1.0.5

Revisit datacube and raster objects for multi-tile access.

## [1.0.4] - 2023-01-27

Tools for L2B handling.

## [1.0.3] - 2023-01-02

Fix for reprojection in any system, especially useful for reprojection in pseudo-Mercator (EPSG:3857) for
visualization with web mapping tools (e.g. OpenStreetMap, Google Maps).

[Unreleased]: https://github.com/Tristanovsk/grstbx/compare/v2.5.0...HEAD
[2.5.0]: https://github.com/Tristanovsk/grstbx/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/Tristanovsk/grstbx/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/Tristanovsk/grstbx/tree/v2.3.0
[2.2.0]: https://github.com/Tristanovsk/grstbx/tree/v2.2.0
[2.1.1]: https://github.com/Tristanovsk/grstbx/tree/v2.1.1
[2.0.3]: https://github.com/Tristanovsk/grstbx/tree/v2.0.3
[1.0.4]: https://github.com/Tristanovsk/grstbx/tree/v1.0.4
[1.0.3]: https://github.com/Tristanovsk/grstbx/tree/v1.0.3
