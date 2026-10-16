# Hyperspectral hGRS products

The L2A products of [hGRS](https://github.com/Tristanovsk/hgrs), the hyperspectral version of GRS for
EnMAP and PRISMA, are opened with {func}`~grstbx.driver.open_hgrs_l2a` (or {func}`~grstbx.subset.open_l2a`),
which splits each NetCDF file as a GRS product: the full-resolution $R_{rs}$ and sunglint, and the
coarse-grid aerosol, water vapor and sunglint as ancillary data. They can then be cropped with
{func}`~grstbx.subset.crop_l2a`, drawn with {func}`~grstbx.utils.plot_rgb` and explored with the viewers
of {mod}`grstbx.visual` or the {doc}`web dashboard <dashboard>`.

## PRISMA, from L1 to L2A

This notebook compares a PRISMA L2A product of hGRS with the PRISMA files given as input to the
processor, on the grid of the L2A product:

- the hGRS L2A product: $R_{rs}$ and sunglint on the full-resolution longitude-latitude grid, aerosol,
  water vapor and sunglint retrieved on the mega-pixels,
- the L1 radiance converted into TOA reflectance and the ASI L2C surface reflectance, read with the
  `hgrs` package and interpolated onto the L2A grid with [xESMF](https://xesmf.readthedocs.io),
- true-colour composites before and after the atmospheric and sunglint correction, maps of the
  retrieved parameters and of the viewing geometry,
- spectra of points of interest from TOA to $R_{rs}$, spectral transect, and comparison of $R_{rs}$ with
  the ASI L2C reflectance, not corrected for the sunglint,
- the interactive viewer {class}`~grstbx.visual.ViewSpectral`.

```python
from grstbx import utils, visual
from grstbx.driver import open_hgrs_l2a

l2a, ancillary = open_hgrs_l2a(l2a_file)       # product and coarse-grid ancillary data
ax = utils.plot_rgb(l2a)                       # true-colour composite on map axes
viewer = visual.ViewSpectral(l2a.Rrs, reproject=True, title='## PRISMA L2A')
viewer.visu()
```

The image is the PRISMA acquisition of 2026-02-06 over three islands of the South Atlantic coast of
Brazil, with strong sunglint. The notebook is rendered with the outputs of its last run (the interactive
viewer cells are not executed). The input data are not distributed with grstbx: set the paths of its
*Input files* cell to run it on your own data.

## EnMAP time series

The [EnMAP tutorial](https://hgrs.readthedocs.io/en/latest/tutorials/grstbx_visualization.html) of the
hGRS documentation explores 14 EnMAP L2A products of Lake Trasimeno (Italy) with grstbx: true-colour
composite and area of interest, crop with the ancillary data, multi-date spectra and time series,
interactive viewers.

```{toctree}
:maxdepth: 1

hgrs_products/grstbx_hgrs_prisma_l1_l2a
```
