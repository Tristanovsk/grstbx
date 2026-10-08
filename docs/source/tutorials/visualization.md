# Interactive visualization

{mod}`grstbx.visual` provides holoviews / datashader / panel viewers to explore datacubes in
Jupyter notebooks. The module is slow to import and is therefore loaded only when it is accessed:

```python
import grstbx

visual = grstbx.visual
```

```{figure} ../../../illustration/grstbx_visual_tool.gif
:alt: animated dashboard of grstbx

{class}`~grstbx.visual.ViewSpectral` dashboard: date and band selection, colormap and color range,
basemap, drawing of areas and points of interest with their spectra.
```

## L2A: remote-sensing reflectance

```python
viewer = visual.ViewSpectral(dc.datacube.Rrs, reproject=True)
viewer.visu()
```

`raster` is a DataArray with dims `(time, wl, y, x)` (`time` is optional). Set `reproject=True` if
the images are not already in web Mercator (EPSG:3857), which is needed to overlay the basemaps.
The initial color range (`minmaxvalues`) and the bounds of its slider (`minmax`) can be adjusted:

```python
viewer = visual.ViewSpectral(raster, reproject=True, minmaxvalues=(0, 0.04), minmax=(0, 0.1))
```

## L2B: water quality parameters

```python
viewer = visual.ViewParam(dc.datacube, reproject=True, minmaxvalues=(0, 4), minmax=(0, 10))
viewer.visu()
```

By default, all the 2D variables (with `x` and `y` dimensions) are proposed; use `params` to
restrict the list.

## Retrieving the drawn geometries

The polygons and points drawn on the maps are stored in holoviews streams and can be converted to
{class}`geopandas.GeoDataFrame` objects, for example to crop the datacube:

```python
aoi = viewer.get_geom(viewer.aoi_stream, crs=raster.rio.crs)    # last polygon drawn
clipped = raster.rio.clip(aoi.geometry.values)

points = viewer.get_points(viewer.poi_stream)                   # ViewSpectral only, EPSG:4326
```
