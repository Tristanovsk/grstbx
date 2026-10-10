# Exploring an L2A image

This notebook walks through a single GRS L2A product, from loading to simple water-quality maps:

- loading of a Zarr or netCDF product with {class}`~grstbx.driver.L2grs` (multiscale pyramid levels),
- decoding of the `flags` bitmask with {class}`~grstbx.masking.Masking`: map of each flag and
  overlay of cloud / land flags on the true-colour image,
- quick-looks: true colour, sunglint, aerosol optical thickness, viewing geometry, per-band maps,
- selection of the water pixels,
- interactive exploration with {class}`~grstbx.visual.ViewSpectral`: multiscale display of the
  Zarr pyramid over a free basemap (CARTO, OpenStreetMap, Esri...), colormaps adapted to water
  reflectance, spectra of drawn points, statistics of $R_{rs}$ over drawn polygons,
- empirical water-quality algorithms: chlorophyll-a (NASA OC2), CDOM absorption
  (Brezonik et al., 2015) and suspended particulate matter (Nechad et al., 2010, 2016), displayed
  with {class}`~grstbx.visual.ViewParam`.

It is rendered with the outputs of its last run, where no geometry was drawn on the interactive
maps (the interactive widgets are not functional in this page). The input image is not distributed
with grstbx: set `workdir` and `DEFAULT_FILE` to run it on your own data. The notebook is
[`notebook/grstbx_l2a_visu.ipynb`](https://github.com/Tristanovsk/grstbx/blob/main/notebook/grstbx_l2a_visu.ipynb).

```{toctree}
:maxdepth: 1

l2a_image/grstbx_l2a_visu
```
