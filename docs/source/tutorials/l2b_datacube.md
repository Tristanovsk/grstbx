# Exploring an L2B datacube

This notebook walks through a year of GRS L2B products (water-quality parameters) over the Berre
lagoon (France), around the HYPERNETS station `BEFR`:

- building of a multi-temporal datacube of the L2B products, cropped to a box centred on the station,
  with {class}`~grstbx.driver.L2grs`, and masking of the non-water / cloudy pixels,
- interactive exploration with {class}`~grstbx.visual.ViewParam`: map of each parameter and date over
  a free basemap, live time series of the pixel under the mouse, time series of drawn points,
  statistics of the parameters over drawn polygons,
- maps of the parameters over one month and monthly composites.

It is rendered with the outputs of its last run (the interactive widgets are not functional in this
page): the time series and polygon statistics are computed over the default points and area of
interest preloaded on the map with `poi=` and `aoi=` (around the BEFR station). The input products are
not distributed with grstbx: set `L2B_DIR` and `DATACUBE_DIR` to run it on your own data. The notebook
is
[`notebook/validation/grstbx_l2b_datacube_visu.ipynb`](https://github.com/Tristanovsk/grstbx/blob/main/notebook/validation/grstbx_l2b_datacube_visu.ipynb).

```{toctree}
:maxdepth: 1

l2b_datacube/grstbx_l2b_datacube_visu
```
