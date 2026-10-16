# Exploring an L2B datacube

This notebook explores a year of GRS L2B products (water-quality parameters) around a site, with
{class}`~grstbx.driver.L2grs` and {class}`~grstbx.visual.ViewParam`:

- list of the L2B products stored as `<site>/YYYY/MM/DD/<product>.nc` and multi-temporal datacube cropped
  to a box centred on the site, cached as a NetCDF file,
- masking of the non-water and cloudy pixels (`mask`), proportion of valid pixels per image and
  selection of the images,
- interactive exploration: map of each parameter and date over a free basemap, time series of the pixel
  under the mouse and of drawn points, statistics of the parameters over drawn polygons,
- maps of the parameters over one month and monthly composites.

```python
import geopandas as gpd
import grstbx
from grstbx import visual

box = grstbx.SpatioTemp().wktbox(lon, lat, width=8000, height=8000)   # box around the site
bbox = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt([box]), crs=4326)
dc = grstbx.L2grs(files)                       # L2B products (NetCDF or Zarr)
dc.get_l2b_datacube(subset=bbox)

viewer = visual.ViewParam(dc.datacube, params=['Chla_OC2nasa', 'SPM_nechad'], reproject=True,
                          cmap='cmo.algae')
viewer.visu()
```

The datacube gathers the Sentinel-2 A/B L2B products (20 m, GRS v2.2.1) of 2024 over the Berre lagoon
(France), in a box of 8 km around the HYPERNETS station `BEFR`. The notebook is rendered with the outputs
of its last run: the interactive widgets are not functional in this page, and the time series and
polygon statistics are computed over the default points and area of interest preloaded on the map with
`poi=` and `aoi=`. The input products are not distributed with grstbx: set `L2B_DIR` and `DATACUBE_DIR`
to run it on your own data. The same datacube can be opened in the {doc}`web dashboard <dashboard>`.

```{toctree}
:maxdepth: 1

l2b_datacube/grstbx_l2b_datacube_visu
```
