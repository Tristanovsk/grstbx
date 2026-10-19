# Utilities

{mod}`grstbx.utils` gathers the helpers used in the post-processing notebooks: geometry, in situ data,
solar irradiance, terrain illumination, plotting and regridding. Heavy dependencies (matplotlib, scipy,
scikit-learn, xesmf) are imported only when the corresponding function is called.

The figures of this page are drawn from GRS products and in situ data that are not distributed with
grstbx: a Sentinel-2 L2A image of the Loire estuary (2017-04-12, tile 30TWT), the L2A products of tile
31TEJ (August 2025) and the HYPERNETS data of the Berre lagoon (see {doc}`validation`). They are produced
by the script
[`utilities/make_figures.py`](https://github.com/Tristanovsk/grstbx/blob/main/docs/source/tutorials/utilities/make_figures.py):
set its paths to reproduce them with your own data.

## Geometry

{meth}`SpatioTemp.wktbox <grstbx.utils.SpatioTemp.wktbox>` returns a rectangle of a given width and
height (m) centred on a point, as a WKT polygon in longitude / latitude, and
{meth}`SpatioTemp.clip_raster <grstbx.utils.SpatioTemp.clip_raster>` cuts a square of side `extent_m`
around a point from an image with a CRS:

```python
from grstbx import SpatioTemp

st = SpatioTemp()
box = st.wktbox(lon, lat, width=5000, height=5000)       # WKT polygon around (lon, lat)
clipped = st.clip_raster(product[['Rrs']], lat, lon, extent_m=5000)   # 5 x 5 km
```

```{figure} utilities/geometry.png
:alt: wktbox and clip_raster on the Loire estuary

5 km box around a point of the Loire river plume on a true-colour composite of the image (left), and the
subset cut by `clip_raster` at full resolution: 250 x 250 pixels of 20 m (right).
```

To cut a product with all its variables and its ancillary data, and export the subset as a GRS product,
use {func}`~grstbx.subset.crop_l2a` (see {doc}`subset_export`).

## In situ data

Read AERONET-OC (version 3) files into a {class}`pandas.DataFrame` with 3-level columns (name, data type,
wavelength):

```python
from grstbx import Data

df = Data().read_aeronet_ocv3('20200101_20201231_Venise.lev15', skiprows=8)
```

Remove wavelength ranges (e.g. absorption bands) from a hyperspectral DataArray or Dataset with a `wl`
dimension ({meth}`Data.remove_wl_dataarray <grstbx.utils.Data.remove_wl_dataarray>`,
{meth}`Data.remove_wl_dataset <grstbx.utils.Data.remove_wl_dataset>`); only the wavelength coordinate is
read, the data stay lazy:

```python
removed = [(683, 695), (755, 770), (810, 835)]   # O2-B, O2-A, water vapour (nm)
Rrs = Data.remove_wl_dataarray(Rrs, wl_to_remove=removed)
```

```{figure} utilities/remove_wl.png
:alt: HYPERNETS spectrum with absorption bands removed

HYPERNETS spectrum of the Berre lagoon (median of one day): the wavelengths of the absorption bands of
O$_2$ (687 and 760 nm) and water vapour (820 nm), where the in situ spectra are noisier, are removed.
```

## Solar irradiance

Extraterrestrial solar irradiance (Thuillier et al., 2003, 0.3 nm resolution, distributed with grstbx),
interpolated at any wavelength in mW m{sup}`-2` nm{sup}`-1`:

```python
from grstbx import Irradiance

irr = Irradiance()
irr.load_F0()                                 # table in irr.F0df (columns wl, F0)
F0 = irr.get_F0(dc.datacube.wl.values)
```

```{figure} utilities/irradiance.png
:alt: extraterrestrial solar irradiance

Extraterrestrial solar irradiance and its values at the central wavelengths of the Sentinel-2 bands.
```

## Terrain illumination

{meth}`Dem.compute_dem_attributes <grstbx.utils.Dem.compute_dem_attributes>` computes the slope, the
aspect and the cosine of the local solar incidence angle from a DEM, useful to identify the terrain
shadows:

$$
\cos\theta_i = \cos\theta_s \cos\beta + \sin\theta_s \sin\beta \cos(\phi_s - \alpha)
$$

with $\theta_s$ the solar zenith angle, $\phi_s$ the solar azimuth, $\beta$ the slope and $\alpha$
the aspect of the terrain (downslope direction, clockwise from North). The DEM must be in a projected
coordinate system (x and y in meter); `z_factor` applies a vertical exaggeration, useful to display
the relief (see {doc}`case_studies`). The GRS L2A products hold the DEM used by the processing (`dem`)
and the mean sun position (attributes `mean_solar_zenith_angle`, `mean_solar_azimuth`):

```python
from grstbx import Dem

sza, azi = product.attrs['mean_solar_zenith_angle'], product.attrs['mean_solar_azimuth']
attrs = Dem.compute_dem_attributes(product.dem, sza, azi)
attrs.shaded.plot.imshow(cmap='Greys_r')
```

```{figure} utilities/terrain.png
:alt: DEM, slope and illumination of tile 31TEJ

DEM of an L2A product of tile 31TEJ (Cevennes and Gulf of Lion) at 80 m, slope, and cosine of the local
incidence angle for the sun position of the acquisition: the slopes facing away from the sun are darker.
With `z_factor=3`, the relief is enhanced for the display.
```

## Plotting

### Maps

{func}`~grstbx.utils.utm_projection` returns the cartopy projection matching the CRS of an image
(UTM zone and hemisphere for Sentinel-2 tiles, plate carrée for longitude-latitude grids such as the
hGRS PRISMA products, web Mercator otherwise), to plot it on map axes.
{func}`~grstbx.utils.plot_rgb` displays a true-colour (or false-colour) composite of $R_{rs}$, with
the outline of an area of interest. Large images are decimated to at most `max_size` pixels per side
(2000 by default), so that a full tile is displayed in a few seconds.

```python
import matplotlib.pyplot as plt
import grstbx

ax = grstbx.plot_rgb(product, aoi=aoi)           # map axes in the projection of the image

proj = grstbx.utm_projection(product.rio.crs)
fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': proj})
grstbx.plot_rgb(product, ax=axs[0], aoi=aoi)                         # 665, 560, 490 nm
grstbx.plot_rgb(product, ax=axs[1], bands=(865, 665, 560), gamma=1)  # false colour
```

```{figure} utilities/maps.png
:alt: true and false colour composites of the Loire estuary

True-colour composite of the Loire estuary with the outline of an area of interest (left), and
false-colour composite with the near-infrared band (right), where the vegetation is red and the turbid
plume stands out against the clear waters.
```

### Matchups

{class}`~grstbx.utils.Plotting` provides matplotlib helpers for matchup scatter plots:
{meth}`~grstbx.utils.Plotting.set_layout` makes the axes square with a 1:1 line, and
{meth}`~grstbx.utils.Plotting.add_stats` draws the linear regression and writes its statistics:

```python
import matplotlib.pyplot as plt
from grstbx import Plotting

fig, ax = plt.subplots()
ax.scatter(insitu, satellite, c=wavelength, cmap='Spectral_r')
Plotting.set_layout(ax)                                 # square axes and 1:1 line
Plotting.add_stats(insitu, satellite, ax, label=True)   # regression, r, rmse, mape, N
```

```{figure} utilities/matchups.png
:alt: matchups of Sentinel-2 and HYPERNETS Rrs

$R_{rs}$ of Sentinel-2 (GRS) against HYPERNETS in the Berre lagoon, bands 443 to 705 nm, coloured by
wavelength. The mape is given as a fraction (0.595 = 59.5 %).
```

The {mod}`grstbx.matchup` module pairs satellite and in situ data and computes more robust agreement
statistics (median symmetric accuracy, signed symmetric bias), see {doc}`validation`.

## Regridding

{meth}`Reproj.regridding <grstbx.utils.Reproj.regridding>` regrids swath products (2D latitude
and longitude) onto a regular grid with [xESMF](https://xesmf.readthedocs.io), installed with the
`regrid` extra (`pip install "grstbx[regrid]"`, xesmf needs ESMF:
`conda install -c conda-forge xesmf`). The interpolation of PRISMA images onto the grid of an hGRS L2A
product is shown in {doc}`hgrs_products`.
