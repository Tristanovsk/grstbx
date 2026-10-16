# Web dashboard

`grstbx-viewer` opens an L2A product in a web dashboard and explores it with
{class}`~grstbx.visual.ViewSpectral`, without writing a notebook. It accepts GRS L2A Zarr stores
(single resolution or multiscale pyramids), GRS L2A NetCDF product folders (`<name>/` holding
`<name>.nc` and `<name>_anc.nc`) and hGRS L2A NetCDF files (EnMAP, PRISMA).

```{figure} ../../../illustration/grstbx_viewer_dashboard.png
:alt: grstbx-viewer dashboard

Sentinel-2 L2A image of the Loire estuary in the dashboard: product selection in the sidebar,
true-colour composite over OpenStreetMap, spectra of the points and under the mouse, export buttons.
```

## Start the dashboard

```console
grstbx-viewer                                      # choose the product in the sidebar
grstbx-viewer /data/S2A_MSIL2AGRS_..._V3.0.1.zarr  # open a product at start-up
grstbx-viewer --root /data/satellite/Sentinel-2    # folder of the file browser
```

The dashboard opens in the web browser (`--no-show` to prevent it). Choose the product by typing its
path in the *Product* field or with the *File browser* of the sidebar (a click on a folder opens it, a
click on a product, marked with a globe, selects it), then click **Open**. The *Display* card sets the
basemap, the initial colormap and the colour range of the single bands; `--basemap`, `--cmap` and
`--no-reproject` (projection of the image, without basemap) set them from the command line.

## Explore the image

The viewer is the one of the notebooks (see {doc}`l2a_image`):

- pan and zoom the map: only the pixels displayed are read, at the screen resolution, from the Zarr
  pyramid levels, so a whole Sentinel-2 tile stays fluid;
- choose the band (or the true-colour composite), the basemap, the colormap and the colour range with
  the widgets;
- hover the map to plot the full-resolution spectrum of the pixel under the mouse (dashed line);
- add points with the **Point Draw** tool (click) to plot their spectra;
- draw polygons with the **Polygon Draw** tool (double-click to start and to close a polygon).

Below the map, the **Export** buttons download the full-resolution spectra of the points
({func}`~grstbx.app.points_spectra`), the statistics of $R_{rs}$ over the valid pixels of each polygon
(median, mean, standard deviation, 5, 25, 75 and 95 % percentiles, number of pixels;
{func}`~grstbx.app.polygons_statistics`) as CSV files, and the points and polygons as GeoJSON files.

## Share the dashboard on a network

The dashboard is a [Panel](https://panel.holoviz.org) application: a Python process reads the products
and computes the images and spectra requested by the browser. The products are therefore read on the
machine running `grstbx-viewer`, which is best started where the data are stored (e.g. on a
processing server), and opened from other computers:

```console
grstbx-viewer --root /data/satellite --address 0.0.0.0 --port 5006 \
              --allow-websocket-origin myserver:5006
```

Each browser session opens its own copy of the product. The basemaps are downloaded by the browser:
the Stadia Maps styles need an API key outside `localhost` (`--stadia-api-key` or the
`STADIA_API_KEY` environment variable, see {func}`~grstbx.visual.set_stadia_api_key`).

From Python, {func}`grstbx.app.serve` starts the same server and {func}`grstbx.app.create_app` returns
the dashboard, to embed it in another Panel application.
