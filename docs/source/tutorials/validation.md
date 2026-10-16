# Validation with in situ data

These notebooks compare GRS products with in situ measurements in the Berre lagoon (France), where the
HYPERNETS station `BEFR` is installed. Both build a multi-temporal datacube of the products over a box
centred on the station ({class}`~grstbx.driver.L2grs`), compute the statistics of the valid pixels of the
box for each image, pair each image with the in situ data acquired within a maximum time lag, and compute
the agreement statistics.

```python
import pandas as pd
import grstbx

# sat_stats: statistics of the box per image; insitu: in situ time series
matchups = grstbx.get_matchups(sat_stats[['median', 'q25', 'q75']], insitu,
                               max_lag=pd.Timedelta('1D'))
metrics = grstbx.matchup_metrics(matchups.insitu, matchups['median'])
grstbx.plot_matchup(ax, matchups.insitu, matchups['median'], 0, vmax)
```

The notebooks are rendered with the outputs of their last run. The input data are not distributed with
grstbx: set the paths of their configuration cell to run them on your own data.

## L2A: remote-sensing reflectance against HYPERNETS

The first notebook evaluates $R_{rs}$ of the GRS L2A products (v2.1.7, 2021-2023) against the HYPERNETS
autonomous hyperspectral radiometer:

- HYPERNETS time series: one NetCDF file built from the individual L2A files of the station, daily
  spectra and O$_2$ absorption bands,
- normalization of the in situ $R_{rs}$, measured at oblique viewing, to nadir viewing,
- datacube of a 1 km box, flags of the images (cirrus, clouds, water) and statistics of the valid pixels,
- matchups within 2 hours, in situ spectra interpolated to the Sentinel-2 wavelengths, spectra of each
  matchup with an RGB thumbnail of the box,
- agreement statistics per band and for all the bands, before and after the normalization to nadir
  viewing: regression, rmse, bias, median symmetric accuracy $\epsilon$ and signed symmetric bias
  $\beta$.

## L2B: chlorophyll-a and suspended matter against the GIPREB samples

The second notebook compares the chlorophyll-a and suspended particulate matter of the GRS L2B products
(v2.2.1, 2017-2024) with the samples of the GIPREB monitoring network:

- GIPREB database: monthly samples at the surface and at the bottom of 10 stations,
- datacube of a 400 m box centred on one station,
- quality control of the satellite pixels (GRS `mask`, distance to the optical water types of Spyrakos
  et al., 2018) and of the images (proportion of valid pixels, homogeneity of the box),
- satellite and in situ time series,
- matchups within one day ({func}`~grstbx.matchup.get_matchups`) and agreement statistics
  ({func}`~grstbx.matchup.matchup_metrics`, {func}`~grstbx.matchup.plot_matchup`).

The notebooks are in
[`notebook/validation/`](https://github.com/Tristanovsk/grstbx/tree/main/notebook/validation).

```{toctree}
:maxdepth: 1

validation/grstbx_l2a_hypernets_matchup
validation/grstbx_l2b_hypernets_matchup
```
