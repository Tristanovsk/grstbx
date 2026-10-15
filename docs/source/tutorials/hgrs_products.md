# Hyperspectral hGRS products

The L2A products of [hGRS](https://github.com/Tristanovsk/hgrs), the hyperspectral version of GRS
for EnMAP and PRISMA, are opened with {func}`~grstbx.driver.open_hgrs_l2a` (or
{func}`~grstbx.subset.open_l2a`), which splits each NetCDF file as a GRS product: the full-resolution
$R_{rs}$ and sunglint, and the coarse-grid aerosol, water vapor and sunglint as ancillary data. They
can then be cropped with {func}`~grstbx.subset.crop_l2a`, drawn with {func}`~grstbx.utils.plot_rgb`
and explored with the viewers of {mod}`grstbx.visual`. Tutorials:

- **PRISMA, from L1 to L2A**: a PRISMA image over the coast of Brazil with strong sunglint; the L1
  TOA reflectance and the ASI L2C reflectance given as input to hGRS are interpolated onto the grid
  of the L2A product, to compare the true-colour composites before and after correction, map the
  retrieved parameters and viewing geometry, and plot the spectra of points of interest from TOA to
  $R_{rs}$, a spectral transect and the comparison with the L2C reflectance. The PRISMA files are read
  with the `hgrs` package and interpolated with [xESMF](https://xesmf.readthedocs.io).
- [**EnMAP time series**](https://hgrs.readthedocs.io/en/latest/tutorials/grstbx_visualization.html)
  (documentation of hGRS): 14 EnMAP L2A products of Lake Trasimeno (Italy), true-colour composite
  and area of interest, crop with the ancillary data, multi-date spectra and time series, interactive
  viewers.

The PRISMA notebook is rendered with the outputs of its last run (the interactive viewer cells are
not executed): the input data are not distributed with grstbx, set the paths of its *Input files*
cell to run it on your own data. The notebook is
[`notebook/visu/grstbx_hgrs_prisma_l1_l2a.ipynb`](https://github.com/Tristanovsk/grstbx/blob/main/notebook/visu/grstbx_hgrs_prisma_l1_l2a.ipynb).

```{toctree}
:maxdepth: 1

hgrs_products/grstbx_hgrs_prisma_l1_l2a
```
