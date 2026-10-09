# Utilities

{mod}`grstbx.utils` gathers helpers used in the post-processing notebooks. Heavy dependencies
(matplotlib, scipy, scikit-learn, xesmf) are imported only when the corresponding function is
called.

## Geometry

```python
from grstbx import SpatioTemp

st = SpatioTemp()
box = st.wktbox(lon, lat, width=1000, height=1000)       # WKT polygon centred on (lon, lat)
clipped = st.clip_raster(raster, lat, lon, extent_m=500)  # square subset around a point
```

## In-situ data

Read AERONET-OC (version 3) files into a {class}`pandas.DataFrame` with 3-level columns (name,
data type, wavelength):

```python
from grstbx import Data

df = Data().read_aeronet_ocv3('20200101_20201231_Venise.lev15', skiprows=8)
```

Remove wavelength ranges (e.g. absorption bands) from a datacube:

```python
Rrs = Data.remove_wl_dataarray(Rrs, wl_to_remove=[(930, 960), (1300, 1500)])
```

## Solar irradiance

Extraterrestrial solar irradiance (Thuillier et al., 2003), interpolated in mW m{sup}`-2` nm{sup}`-1`:

```python
from grstbx import Irradiance

irr = Irradiance()
irr.load_F0()
F0 = irr.get_F0(dc.datacube.wl.values)
```

## Terrain illumination

{meth}`Dem.compute_dem_attributes <grstbx.utils.Dem.compute_dem_attributes>` computes the slope and
the cosine of the local solar incidence angle from a DEM, useful to identify the terrain shadows:

$$
\cos\theta_i = \cos\theta_s \cos\beta + \sin\theta_s \sin\beta \cos(\phi_s - \alpha)
$$

with $\theta_s$ the solar zenith angle, $\phi_s$ the solar azimuth, $\beta$ the slope and $\alpha$
the aspect of the terrain (downslope direction, clockwise from North). The DEM must be in a projected
coordinate system (x and y in meter); `z_factor` applies a vertical exaggeration, useful to display
the relief (see {doc}`case_studies`).

```python
from grstbx import Dem

attrs = Dem.compute_dem_attributes(datacube.dem, sza=35., azi=150.)
attrs.shaded.plot.imshow()
```

## Plotting

{class}`~grstbx.utils.Plotting` provides matplotlib helpers for matchup scatter plots:

```python
import matplotlib.pyplot as plt
from grstbx import Plotting

fig, ax = plt.subplots()
ax.scatter(insitu, satellite)
Plotting.set_layout(ax)                              # square axes and 1:1 line
Plotting.add_stats(insitu, satellite, ax, label=True)  # regression, r, rmse, mape, N
```

## Regridding

{meth}`Reproj.regridding <grstbx.utils.Reproj.regridding>` regrids swath products (2D latitude
and longitude) onto a regular grid with [xESMF](https://xesmf.readthedocs.io), installed with the
`regrid` extra (`pip install "grstbx[regrid]"`).
