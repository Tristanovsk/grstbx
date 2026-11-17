# Validation with AERONET-OC

{mod}`grstbx.validation` validates the remote sensing reflectance $R_{rs}$ of the GRS L2A products with the
AERONET-OC measurements, on a dataset of Sentinel-2 images cut around the AERONET-OC stations, and
{mod}`grstbx.validation_app` is a dashboard to analyse the matchups interactively. The chain:

1. **dataset**: Sentinel-2 L1C subsets of 5 km x 5 km around the stations, with a static STAC catalogue, in an S3
   bucket;
2. **GRS** on the subsets, one process per image;
3. **matchup table**: the L2A pixels around each station paired with the nearest AERONET-OC measurement, one row
   per image and band;
4. **quality criteria and statistics**, by band, station...;
5. **dashboard**.

```python
from grstbx import validation as val

items = val.read_items(stations=['AAOT'])            # subsets of the STAC catalogue
runs = val.process_subsets(items, workers=2)          # GRS on the subsets without L2A product
table = val.matchup_table(items)                      # one row per image and band
val.save_table(table, '/data/satellite/validation/matchups/grs_3.1.1.csv')
ok = val.quality(table)                               # default criteria: Level 2.0, 1 h, 50 % valid, CV < 0.2
val.metrics_table(table[ok], by='wl')                 # statistics by band
val.dashboard(table)                                  # dashboard, in a notebook
```

or from the command line (`grstbx-validation`, installed with grstbx):

```bash
grstbx-validation grs --stations AAOT --workers 2
grstbx-validation matchups -o /data/satellite/validation/matchups/grs_3.1.1.csv
grstbx-validation dashboard /data/satellite/validation/matchups/grs_3.1.1.csv --show
```

The notebooks of the chain are in
[`notebook/validation_image_subset`](https://github.com/Tristanovsk/grstbx/tree/main/notebook/validation_image_subset):
`aeronet_oc_s2_l1c_subsets.ipynb` (the dataset) and `grs_aeronet_oc_validation.ipynb` (GRS, matchups, statistics
and dashboard). The data are not distributed with grstbx.

## Dataset of Sentinel-2 subsets

The script `notebook/validation_image_subset/aeronet_oc_s2_subsets.py` makes the dataset:

| item | content |
|---|---|
| stations | the AERONET-OC sites of the in situ validation database; the sites at the same position are one station (e.g. Venise and AAOT) |
| images | the Sentinel-2 overpasses (CDSE catalogue, last processing) with an AERONET-OC measurement (Level 1.5 or 2.0) within 2 h; one tile per datatake (best coverage of the box, then an image already on the machine, then the UTM zone of the station) |
| subsets | 251 x 251 pixels at 20 m (5 km) centred on the station, which falls in the central pixel |
| format | product of GRSdriver (`Sentinel2Driver`): TOA reflectance of the 13 bands, viewing angles per band and detector (computed at 60 m), sun angles, spectral responses; NetCDF with the bands and angles packed in int16, about 1.3 MB; read directly by GRS (`grs <subset>.nc`), without recomputing the angles |
| source | the images of the machine read in place; the others downloaded from CDSE, cut and deleted one by one (or read in place in the CDSE bucket, `--source remote`) |

Layout of the bucket `s3://aeronet-oc` (content in `/data/satellite/validation`, served with the other buckets by
`grstbx-s3-bucket`, see {doc}`s3_stac`):

```
sentinel-2/l1c/<station>/YYYY/MM/DD/<name>.nc      subset (<name> = <Sentinel-2 name>_<station>)
sentinel-2/l1c/<station>/YYYY/MM/DD/<name>.json    STAC Item, written last
sentinel-2/l1c/<station>/YYYY/MM/DD/<name>_ql.png  quicklook (TOA RGB, station marked)
sentinel-2/l2a/<station>/YYYY/MM/DD/<name>/        GRS L2A products of the subsets, with their STAC Item
stations.geojson                                   stations, AERONET-OC sites, boxes, number of subsets
stac/catalog.json                                  static STAC catalogue, collection s2-l1c-aeronet-oc
```

The STAC Items carry the AERONET-OC measurements around the overpass (`aeronet_oc:station`, `aeronet_oc:sites`,
`aeronet_oc:n_obs`, `aeronet_oc:n_obs_lev20`, `aeronet_oc:dt_min`: time of the nearest measurement minus
overpass, in minutes), with the usual properties (`eo:cloud_cover`, `grid:code`, `proj:*`, `view:*`): the
catalogue is searched with {func}`grstbx.app.search_items`, as in the dashboard of grstbx, and read by
{func}`~grstbx.validation.read_items`.

## GRS on the subsets

{func}`~grstbx.validation.process_subsets` runs GRS on the subsets without L2A product, `workers` images at a
time, one process per image (`grstbx-validation grs-one`): the memory is released after each image (about 1 to 2
GB per image) and a crash does not stop the others. Each L2A product is written in its own folder with the
quicklooks of GRS and a STAC Item ({func}`~grstbx.validation.run_grs`); the CAMS file is that of the day if it
exists, else that of the month ({func}`~grstbx.validation.cams_file`). The products already there are reused, a
failure leaves a `<name>.err` file (retried with `retry_failed=True`), an image without water a
`<name>_no_water.txt` marker.

GRS reads the surface pressure `sp` of the CAMS files, which the CAMS global forecasts of the ADS do not provide
from January 2015 to May 2016 (only the mean sea level pressure `msl`): GRS fails on the images of this period
(`.err` file: `'Dataset' object has no attribute 'sp'`). The script
`notebook/validation_image_subset/fill_cams_surface_pressure.py` completes the monthly files of this period with
`sp` derived from their `msl` and 2 m temperature with the barometric formula, and the effective orography of
the CAMS model estimated from the forecasts of June 2016 to May 2017 (the original files are kept in
`original_without_sp/`). Compared with the `sp` of the CAMS forecasts of December 2016 and March 2017 (not used
for the orography), the derived pressure is within 0.01 hPa over the sea and low coasts (95 % of the cells),
where the AERONET-OC stations are, and within about 15 hPa over the mountains above 1000 m.

## Matchup protocol

{func}`~grstbx.validation.matchup_table` builds the table, with the protocol of the validation of GRS:

| step | default |
|---|---|
| satellite window | ring of pixels 4 to 7 pixels (80 to 140 m) around the station (Chebyshev distance), without the platform and its shadow; or a box (`window='box'`) |
| valid pixels | water (`mask == 0`), finite $R_{rs}$ in all the bands, not brighter than the median of the window at 865 nm by more than 8 10{sup}`-4` sr{sup}`-1` (platform, boats) |
| satellite value | median of the valid pixels per band, with their mean and standard deviation; median AOT(550), sunglint `BRDFg`, wind, angles |
| in situ data | AERONET-OC measurements of the validation database ({func}`~grstbx.validation.read_aeronet_oc`): `Rrs_insitu` = Lwn / F0 without bidirectional correction (directional, as measured: closest to the directional $R_{rs}$ of GRS), `Rrs_insitu_fQ` (f/Q, Morel et al. 2002), `Rrs_insitu_IOP` (Lee et al. 2011), AOT(550), wind, geometry |
| pairing in time | nearest measurement within 3 h, the nearest Level 2.0 one if any, else the nearest Level 1.5 one |
| pairing in wavelength | nearest AERONET-OC band within 15 nm ({func}`~grstbx.validation.band_pairs`), e.g. 665 -> 667, 783 -> 779 nm |

All the images with an L2A product stay in the table: the quality criteria are columns (`lag_min`, `level`,
`frac_valid`, `cv560`, `aot550_sat`, `BRDFg`...) applied afterwards by {func}`~grstbx.validation.quality`, so
that they can be changed without extracting the products again:

| criterion | default |
|---|---|
| lag | 1 h at most (Zibordi et al., 2009) |
| in situ level | Level 2.0 and 1.5 (`levels`); Level 2.0 only in the dashboard at start |
| valid fraction of the window | 50 % at least |
| homogeneity | coefficient of variation of $R_{rs}(560)$ in the window below 0.2 (Bailey and Werdell, 2006) |
| AOT(550), sunglint | no limit (`max_aot`, `max_glint`) |

{func}`~grstbx.validation.metrics_table` gives the statistics of {func}`~grstbx.matchup.matchup_metrics` overall
and by group (band, station, both...): regression (slope, intercept, $r$), RMSE, bias, MAE, median symmetric
accuracy $\epsilon$ (MdSA) and signed symmetric percentage bias $\beta$ (SSPB, Morley et al., 2018), MedAPE and
MAPE, and the RMSE, bias and MAE of $\log_{10} R_{rs}$ (`rmse_log`, `bias_log`, `mae_log`, as the atmospheric
correction intercomparisons ACIX).

## Dashboard

```python
val.dashboard(table)                                      # notebook: sidebar and tabs in a row
```

```bash
grstbx-validation dashboard grs_3.1.1.csv [grs_3.2.0.csv ...] --show   # web application, http://localhost:5007
```

The sidebar selects the matchups: stations, bands, normalisation of the in situ $R_{rs}$, AERONET-OC level,
platforms, period, GRS versions (when several tables are given, to compare versions) and the quality criteria
(lag, valid fraction, CV, AOT, sunglint); the selected matchups can be downloaded as CSV. The views follow the
selection:

| tab | content |
|---|---|
| Overview | map of the stations (size and colour: images with matchups), statistics by band and by station |
| Scatter | GRS against AERONET-OC per band, or all the bands together: error bars (standard deviation of the window), 1:1 and regression lines, statistics, colour by band, station, lag, AOT, sunglint, level, platform or version, hover with the station, date, lag and quality of the window, logarithmic axes |
| Statistics | tables by band, by station and by station and band; heat map of a statistic by station and band |
| Errors | residuals (GRS - AERONET-OC) and relative residuals by band, by station, and their distributions |
| Spectra | the matchups in a table; spectrum of the selected one (GRS median +- standard deviation, AERONET-OC) with the quicklooks of its L2A product (L1C RGB, L2A RGB, sunglint) |
| Time series | residuals in time, per band |
| Drivers | residuals against AOT (GRS or AERONET-OC), sunglint, sun and view zenith, relative azimuth, wind, lag, valid fraction and CV of the window, cloud cover, in situ $R_{rs}$ or date |

The figures below show the first AAOT matchups of January 2017 (two images, Level 2.0 and 1.5, with relaxed
criteria), an illustration of the views rather than a validation result.

```{figure} aeronet_oc_validation/dashboard_overview.png
:alt: dashboard of the matchups, Overview tab

Overview: selection and quality criteria in the sidebar, map of the stations, statistics.
```

```{figure} aeronet_oc_validation/dashboard_scatter.png
:alt: scatter plot of GRS against AERONET-OC Rrs, all bands

Scatter, all the bands: 1:1 line (dashed), regression line (red) and statistics.
```

```{figure} aeronet_oc_validation/dashboard_spectra.png
:alt: spectrum of a matchup with the quicklooks of the L2A product

Spectra: GRS median and standard deviation of the ring around the station, AERONET-OC measurement 16 min
before the overpass, quicklooks of the L2A product.
```
