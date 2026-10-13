# Exploring an L2A image

This notebook walks through a single GRS L2A product, from loading to simple water-quality maps:

- loading of a Zarr or netCDF product with {class}`~grstbx.driver.L2grs` (multiscale pyramid levels),
- interactive exploration with {class}`~grstbx.visual.ViewSpectral`: multiscale display of the
  Zarr pyramid over a free basemap (CARTO, OpenStreetMap, Esri...), colormaps adapted to water
  reflectance, live spectrum of the pixel under the mouse, spectra of drawn points, statistics of
  $R_{rs}$ over drawn polygons,
- empirical water-quality algorithms: chlorophyll-a (NASA OC2), CDOM absorption
  (Brezonik et al., 2015) and suspended particulate matter (Nechad et al., 2010, 2016), displayed
  with {class}`~grstbx.visual.ViewParam` (same colormap picker as `ViewSpectral`).

It is rendered with the outputs of its last run (the interactive widgets are not functional in this
page): the spectra, polygon statistics and water-quality maps are computed over the default points
and area of interest preloaded on the map with `poi=` and `aoi=` (Loire river plume). The input
image is not distributed with grstbx: set `workdir` and `DEFAULT_FILE` to run it on your own data.
The notebook is
[`notebook/visu/grstbx_l2a_visu.ipynb`](https://github.com/Tristanovsk/grstbx/blob/main/notebook/visu/grstbx_l2a_visu.ipynb).

```{toctree}
:maxdepth: 1

l2a_image/grstbx_l2a_visu
```
