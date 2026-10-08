# Multi-temporal datacubes

{class}`~grstbx.driver.L2grs` opens a list of GRS products and concatenates them along a `time`
dimension into `L2grs.datacube`, a lazy (dask-backed) {class}`xarray.Dataset`.

## Area of interest

The images are cropped to the bounding box of a {class}`geopandas.GeoDataFrame`. A box of given
size centred on a point can be built with {meth}`SpatioTemp.wktbox <grstbx.utils.SpatioTemp.wktbox>`:

```python
import geopandas as gpd
import grstbx

lon, lat = 3.6, 43.4                 # centre of the box
width, height = 16000, 16000         # size of the box in m

box = grstbx.SpatioTemp().wktbox(lon, lat, width=width, height=height)
aoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt([box]), crs=4326)
```

## L2A datacube

```python
dc = grstbx.L2grs(files)
dc.get_l2a_datacube(subset=aoi)
dc.datacube
```

For each image, the proportion of pixels raised for each flag of the bitmask is added as a
`flag_<name>` variable (dims: `time`, see {meth}`~grstbx.driver.L2grs.get_flag_stats`). Images
whose `flag_nodata` proportion exceeds `nodata_thresh` (default: 0.5) are skipped. If no image is
kept, `dc.no_product` is set to `True` and no datacube is created.

Other options:

- `reproject=True, epsg_out=3857`: reproject the images, e.g. into web Mercator to overlay
  basemaps in the viewers,
- `FLAG_NAME='flags'`: name of the bitmask variable.

The cloud-free dates can then be selected from the flag statistics:

```python
clear = dc.datacube.flag_cloud < 0.1        # flag names depend on the GRS version
datacube = dc.datacube.sel(time=clear)
```

## L2B datacube

```python
dc = grstbx.L2grs(l2b_files)
dc.get_l2b_datacube(subset=aoi, var='Chla_OC2nasa')
```

A `valid_pix_prop` variable gives, for each date, the proportion of valid pixels of `var`; images
without any valid pixel are skipped. The datacube is sorted by time.

## Multiscale Zarr stores

For pyramids (see {ref}`products <pyramids>`), the resolution level
is chosen when creating the driver; a coarse level is convenient to explore long time series
quickly:

```python
dc = grstbx.L2grs(files, level=2)    # 4x coarser than full resolution
dc.get_l2b_datacube()
```

## Dask chunks

The images are opened with chunks of 1000 x 1000 pixels and all the wavelengths in one chunk. The
chunk sizes can be changed before loading:

```python
dc = grstbx.L2grs(files)
dc.xchunk = dc.ychunk = 2048
```

When an area of interest is given, the cropped images are loaded in memory.
