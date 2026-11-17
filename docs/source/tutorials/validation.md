# Validation with in situ data

The validation of the GRS L2A products with the AERONET-OC stations, on a dataset of Sentinel-2 subsets and with a
dashboard of the matchups, is described in {doc}`aeronet_oc_validation`.

The notebooks are rendered with the outputs of their last run. The input data are not distributed with
grstbx: set the paths of their configuration cell to run them on your own data.

## L2A: remote-sensing reflectance against HYPERNETS

The first notebook evaluates $R_{rs}$ of the GRS L2A products (v3.1.1) against the HYPERNETS (WATERHYPERNET)
autonomous hyperspectral radiometers of the Berre lagoon (`BEFR`) and of the Gironde estuary (`MAFR`), with the
chain of the AERONET-OC validation ({mod}`grstbx.validation`):

- HYPERNETS measurements of the in situ validation database (QC-approved), normalization to nadir viewing,
- Sentinel-2 L1C subsets of 5 km x 5 km centred on the stations (not the full tiles), GRS on the subsets
  ({func}`~grstbx.validation.process_subsets`),
- matchup table ({func}`~grstbx.validation.matchup_table`): ring of water pixels around the station, nearest
  in situ spectrum convolved with the spectral response functions of the Sentinel-2 bands
  ({func}`~grstbx.validation.read_hypernets`),
- quality criteria, agreement statistics by band and station ({func}`~grstbx.validation.metrics_table`), before
  and after the normalization to nadir viewing, and dashboard of the matchups.

```python
import functools
from grstbx import validation as val

reader = functools.partial(val.read_hypernets, srf=val.band_srf(subset), dir_norm=dir_norm)
table = val.matchup_table(items, root=root, reader=reader, max_lag='2h')
val.metrics_table(table[val.quality(table, levels=None)], by='wl', x='Rrs_insitu_nadir')
```

## L2B: chlorophyll-a and suspended matter against the GIPREB samples

The second notebook builds a multi-temporal datacube of the products over a box centred on a station
({class}`~grstbx.driver.L2grs`), computes the statistics of the valid pixels of the box for each image, pairs each
image with the in situ data acquired within a maximum time lag, and computes the agreement statistics:

```python
import pandas as pd
import grstbx

# sat_stats: statistics of the box per image; insitu: in situ time series
matchups = grstbx.get_matchups(sat_stats[['median', 'q25', 'q75']], insitu,
                               max_lag=pd.Timedelta('1D'))
metrics = grstbx.matchup_metrics(matchups.insitu, matchups['median'])
grstbx.plot_matchup(ax, matchups.insitu, matchups['median'], 0, vmax)
```

It compares the chlorophyll-a and suspended particulate matter of the GRS L2B products
(v2.2.1, 2017-2024) with the samples of the GIPREB monitoring network:

- GIPREB database: monthly samples at the surface and at the bottom of 10 stations,
- datacube of a 400 m box centred on one station,
- quality control of the satellite pixels (GRS `mask`, distance to the optical water types of Spyrakos
  et al., 2018) and of the images (proportion of valid pixels, homogeneity of the box),
- satellite and in situ time series,
- matchups within one day ({func}`~grstbx.matchup.get_matchups`) and agreement statistics
  ({func}`~grstbx.matchup.matchup_metrics`, {func}`~grstbx.matchup.plot_matchup`).

The notebooks are in
[`docs/source/tutorials/validation/`](https://github.com/Tristanovsk/grstbx/tree/main/docs/source/tutorials/validation).

```{toctree}
:maxdepth: 1

validation/grstbx_l2a_hypernets_matchup
validation/grstbx_l2b_hypernets_matchup
```
