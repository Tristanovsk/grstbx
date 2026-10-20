# Case studies

Applications of grstbx to time series of GRS L2A and L2B images. The notebooks are rendered with the outputs of
their last run. The input data are not distributed with grstbx: set the paths of their configuration cell
to run them on your own data.

```python
import grstbx

dc = grstbx.L2grs(files)
dc.get_l2a_datacube(subset=aoi)
relief = grstbx.Dem.compute_dem_attributes(dc.datacube.dem, sza, azi)   # slope, illumination
clouds = grstbx.Masking(dc.datacube.isel(time=0)).get_mask(cloud_p06=True)
```

## Clear Lake (California): multi-temporal RGB over shaded relief

The first notebook explores a spring time series of Clear Lake, a large shallow eutrophic lake
surrounded by mountains and subject to recurrent cyanobacteria blooms:

- datacube of the lake from the Sentinel-2 tile 10SEJ (22 images, April-May 2024, 20 m; about 20 GB of
  memory),
- cloud cover and proportion of valid water pixels of each image,
- shaded relief computed from the DEM for the sun position of each acquisition
  ({meth}`~grstbx.utils.Dem.compute_dem_attributes`), under RGB composites of $R_{rs}$ at 705, 560 and
  490 nm, where the blooms appear in red and the terrain shadows near the shores are visible,
- pixel classification flags ({class}`~grstbx.masking.Masking`),
- $R_{rs}$ spectra of an area of the main basin for each date.

## Bagré reservoir (Burkina Faso): Sentinel-2 / in situ SPM matchups

The second notebook evaluates the retrieval of the suspended particulate matter (SPM) in a Sahelian
reservoir loaded with mineral particles:

- datacube of a 2 km box around the Kaporé station (tile 30PYT, 2018-2022), image quality from the
  flags and water pixels of a 200 m disk around the station,
- matchups with the in situ SPM within ±2 days,
- $R_{rs}$ and band ratios against SPM, calibration of a local band-ratio model with leave-one-out
  errors,
- comparison with SPM algorithms from the literature (Nechad et al., 2010; Han et al., 2016; OBS2CO),
- SPM time series and maps.

## Lac de Sainte-Croix (France): L2B water-quality time series

The third notebook follows the water quality of a clear Alpine-foothills reservoir from the L2B products
of full Sentinel-2 tiles:

- products of tile 31TGJ (January 2025 - April 2026) listed from their names
  ({func}`~grstbx.app.find_products`), and selection of the relative orbit that sees the lake, from a
  few products only: the images of the two other orbits have no data over the lake and are not opened,
- datacube of the box around the lake with the variables studied only
  ({meth}`~grstbx.driver.L2grs.get_l2b_datacube` with `subset=` and `variables=`), lake pixels and
  images with enough valid pixels,
- optical water types (Spyrakos et al., 2018),
- maps, per-pixel median and variability, time series and seasonal cycle of the chlorophyll-a,
  turbidity and Secchi disk depth: low chlorophyll-a all year round, turbidity peaks and lower
  transparency in summer,
- interactive exploration with {class}`~grstbx.visual.ViewParam`.

The notebooks are in
[`docs/source/tutorials/case_studies/`](https://github.com/Tristanovsk/grstbx/tree/main/docs/source/tutorials/case_studies).

```{toctree}
:maxdepth: 1

case_studies/grstbx_rgb_dem_multitemp
case_studies/grstbx_l2a_datacube_matchup
case_studies/grstbx_l2b_sainte_croix
```
