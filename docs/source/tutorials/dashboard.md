# Web dashboard

`grstbx-viewer` opens GRS products in a web dashboard, without writing a notebook:

- **L2A** images are displayed with {class}`~grstbx.visual.ViewSpectral` (bands, true-colour composite,
  spectra);
- **L2B** images are displayed with {class}`~grstbx.visual.ViewParam` (water-quality parameters, time
  series).

It opens a single image, a **time series** of products, or a **datacube**:

| input | example |
|---|---|
| GRS L2A or L2B Zarr store (single resolution or multiscale pyramid) | `S2A_MSIL2AGRS_..._V3.0.1.zarr` |
| GRS L2A NetCDF product folder (`<name>/` holding `<name>.nc` and `<name>_anc.nc`) | `S2B_MSIL2A_..._V2.2.1/` |
| GRS L2B NetCDF file | `S2A_MSIL2B_....nc` |
| hGRS L2A NetCDF file (EnMAP, PRISMA) | `PRS_L2A_hgrs_..._V1.1.2.nc` |
| several products: paths or glob patterns, one per line | `/data/L2B/BEFR/2024/*/*/*.nc` |
| folder of products, e.g. the tree `<tile>/YYYY/MM/DD/<product>` of the GRS batches | `/data/L2A_zarr/31TEJ` |
| NetCDF datacube with a `time` dimension (e.g. saved by {class}`~grstbx.driver.L2grs`) | `BEFR_2017-01-01_2024-12-31_H03.nc` |

The products of a time series can be on different grids (e.g. subsets, Zarr and NetCDF mixed) but must
be of the same level (L2A or L2B) and in the same CRS (UTM zone).

```{figure} ../../../illustration/grstbx_viewer_dashboard.png
:alt: grstbx-viewer dashboard, L2A image

Sentinel-2 L2A image of the Loire estuary over OpenStreetMap: SWIR composite (2190, 865 and 665 nm)
chosen in the *RGB composite* card of the sidebar, spectra of the points and under the mouse, export
buttons.
```

```{figure} ../../../illustration/grstbx_viewer_dashboard_l2b.png
:alt: grstbx-viewer dashboard, L2B time series

Time series of 95 Sentinel-2 L2B products over the Berre lagoon (2024), opened with a glob pattern:
chlorophyll-a (OC2) of the first date, the invalid pixels hidden.
```

## Start the dashboard

```console
grstbx-viewer                                        # choose the products in the sidebar
grstbx-viewer /data/S2A_MSIL2AGRS_..._V3.0.1.zarr    # open a product at start-up
grstbx-viewer "/data/L2B/BEFR/2024/*/*/*.nc"         # time series (quote the glob pattern)
grstbx-viewer /data/L2A_zarr --tile 31TEJ            # time series of a tile, from a folder
grstbx-viewer /data/L2B/30TWT --start 2022-01-01 --end 2022-12-31
grstbx-viewer --root /data/satellite/Sentinel-2      # folder of the file browser
```

The dashboard opens in the web browser (`--no-show` to prevent it). Type the paths or glob patterns of
the products in the *Products* field, one per line, or add them with the *File browser* of the sidebar:
a click on a folder opens it, a click on a product (marked with a globe) adds it to the list, and *Add
all the products* adds those of the folder. Then click **Open**. The *Options* card sets, before opening, the basemap,
the initial colormap and colour ranges, and whether the invalid pixels of the L2B products
(`mask != 0`) are hidden; `--basemap`, `--cmap` and `--no-reproject` (projection of the images, without
basemap) set them from the command line.

Once the products are open, two cards of the sidebar adjust the display:

- **Contrast and brightness**: on the RGB composite, they act on the stretched values; on a single band
  or an L2B parameter, they narrow (contrast > 1) and shift (brightness) the colour range set with the
  range slider. *Reset* restores them.
- **RGB composite (L2A)**: band combination (true colour, false colour with the NIR band for the
  vegetation, red edge for the phytoplankton, SWIR for the land / water contrast and the clouds; the
  combinations are proposed when the sensor has the bands), or any band of the sensor for the red, green
  and blue channels, and the gamma applied to $R_{rs}$ to enhance the dark water pixels. The colour
  stretch (2-98 % percentiles) is computed for the bands selected.

## Time series from folders

A folder adds all the products below it, e.g. the tree `<tile>/YYYY/MM/DD/<product>` written by the
GRS batches: the root of the tree, a tile, a year or a month folder. The products are recognized from their
names ({func}`~grstbx.app.find_products`): GRS L2A product folders `<name>/<name>.zarr` (GRS >= 3) or
`<name>/<name>.nc` (GRS 2), Zarr stores and NetCDF files. Add a folder by typing its path in the *Products*
field, or with *Add this folder (time series)* in the file browser.

The **Time series** card of the sidebar lists the products of the folders (*Scan the products*: tiles,
levels, number of products and period), and selects:

- the **tile**: the products of a time series must be in one UTM zone; when the folders hold several
  zones (e.g. tiles 31TEJ and 32TQR), the dashboard asks to choose one (`--tile` from the command line);
- the **period** (*From*, *To*; `--start`, `--end`);
- the **last processing of each acquisition** (default): when an acquisition was processed several times,
  only the latest processing is kept, a release being preferred to a `dev` version
  (`--all-processings` keeps them all).

The products are read only when they are displayed ({func}`~grstbx.app.open_products`): a time series of
hundreds of products opens in a second, and each date is read when it is selected (about 0.5 s for a Zarr
pyramid, 2 s for a NetCDF tile). The time series under the mouse reads one block of each date: for more
than 30 dates, *Live time series* (L2B) is off at opening, tick it to turn it on, or draw points.

## Explore the images

The viewers are the ones of the notebooks (see {doc}`l2a_image` and {doc}`l2b_datacube`):

- pan and zoom the map: only the pixels displayed are read, at the screen resolution, from the Zarr
  pyramid levels, so whole Sentinel-2 tiles stay fluid;
- choose the date and the band (L2A) or the parameter (L2B), the basemap, the colormap and the colour
  range with the widgets;
- hover the map to plot the full-resolution spectrum (L2A) or time series (L2B) of the pixel under the
  mouse (dashed line);
- add points with the **Point Draw** tool (click) to plot their spectra or time series;
- draw polygons with the **Polygon Draw** tool (double-click to start and to close a polygon).

Below the map, the **Export** buttons download, for all the dates:

- the full-resolution spectra or parameters of the points ({func}`~grstbx.app.points_values`): one row
  per point and date, one column per wavelength or parameter;
- the statistics over the valid pixels of each polygon (median, mean, standard deviation, 5, 25, 75 and
  95 % percentiles, number of pixels; {func}`~grstbx.app.polygons_statistics`): one row per polygon,
  date and statistic;
- the points and polygons as GeoJSON files.

The values are read from the products when the button is clicked: for long time series of NetCDF
files, this takes a few tens of seconds (e.g. about 30 s for 95 L2B subsets). A NetCDF datacube smaller
than 1 GB is loaded in memory when opened, so that its time series and exports are fast.

## Share the dashboard on a network

The dashboard is a [Panel](https://panel.holoviz.org) application: a Python process reads the products
and computes the images, spectra and time series requested by the browser. The products are therefore
read on the machine running `grstbx-viewer`, which is best started where the data are stored (e.g. on a
processing server), and opened from other computers:

```console
grstbx-viewer --root /data/satellite --address 0.0.0.0 --port 5006 \
              --allow-websocket-origin myserver.mydomain.fr:5006
```

The dashboard accepts the connections from `localhost`, `127.0.0.1` and the host name of the machine;
add the other names used to reach it with `--allow-websocket-origin` (otherwise the page stays empty and
the server logs *Refusing websocket connection*). Each browser session opens its own copy of the
products. The basemaps are downloaded by the browser: the Stadia Maps styles need an API key outside
`localhost` (`--stadia-api-key` or the `STADIA_API_KEY` environment variable, see
{func}`~grstbx.visual.set_stadia_api_key`).

From Python, {func}`grstbx.app.serve` starts the same server, {func}`grstbx.app.create_app` returns the
dashboard (to embed it in another Panel application) and {func}`grstbx.app.open_products` opens products
for the viewers:

```python
from grstbx import app, visual

data, kind, params, title = app.open_products('/data/L2B/BEFR/2024/*/*/*.nc')
viewer = visual.ViewParam(data, params=params, reproject=True, title=title)
viewer.visu()
```
