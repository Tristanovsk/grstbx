# Exploring an L2A image

This notebook explores a single GRS L2A product, from loading to simple water-quality maps, with
{class}`~grstbx.driver.L2grs` and {class}`~grstbx.visual.ViewSpectral`:

- loading of a Zarr or NetCDF product into a datacube, at full resolution or at a coarser level of the
  Zarr pyramid,
- interactive exploration of $R_{rs}$ over a free basemap (CARTO, OpenStreetMap, Esri...): only the
  pixels displayed are read, at screen resolution, from the Zarr pyramid; true-colour composite or
  single bands with colormaps adapted to water reflectance,
- spectrum of the pixel under the mouse, full-resolution spectra of drawn points and statistics of
  $R_{rs}$ over drawn polygons,
- empirical water-quality algorithms over the last polygon: chlorophyll-a (NASA OC2), CDOM absorption
  (Brezonik et al., 2015) and suspended particulate matter (Nechad et al., 2010, 2016), displayed with
  {class}`~grstbx.visual.ViewParam`.

```python
import grstbx
from grstbx import visual

dc = grstbx.L2grs([file], level=0)              # Zarr store or NetCDF product folder
dc.get_l2a_datacube(nodata_thresh=1.)
l2a_prod = dc.datacube.isel(time=0)

viewer = visual.ViewSpectral(l2a_prod.Rrs, reproject=True, cmap='cmo.turbid')
viewer.visu()
points = viewer.get_points(crs=l2a_prod.rio.crs)   # points drawn on the map
```

The image is the Sentinel-2B acquisition of 2020-04-11 over the Loire estuary and the Bay of Biscay
(tile 30TWT, GRS Zarr product). The notebook is rendered with the outputs of its last run: the
interactive widgets are not functional in this page, and the spectra, polygon statistics and
water-quality maps are computed over the default points and area of interest preloaded on the map with
`poi=` and `aoi=` (Loire river plume). The input image is not distributed with grstbx: set `workdir`
and `DEFAULT_FILE` to run it on your own data. The same exploration without notebook is given by the
{doc}`web dashboard <dashboard>`.

```{toctree}
:maxdepth: 1

l2a_image/grstbx_l2a_visu
```
