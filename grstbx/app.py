"""
Web dashboard to open a GRS (or hGRS) L2A product and explore it with :class:`~grstbx.visual.ViewSpectral`.

The dashboard is a `Panel <https://panel.holoviz.org>`_ application served by a Python process, which
reads the pixels displayed on the screen and the spectra under the mouse: start it with

.. code-block:: console

    grstbx-viewer                                   # empty dashboard, choose the product in the sidebar
    grstbx-viewer /data/S2A_MSIL2AGRS_..._V3.0.1.zarr --root /data/satellite
    grstbx-viewer --address 0.0.0.0 --port 5006 --allow-websocket-origin myserver:5006

or from Python with :func:`serve`. The products are read on the machine running the server: with
``--address 0.0.0.0``, the dashboard is reachable from other computers of the network.
"""

import argparse
import io
import os

import numpy as np
import pandas as pd
import panel as pn

from .driver import is_hgrs, is_zarr

ACCEPTED = ('GRS L2A Zarr store (`*.zarr`), GRS L2A NetCDF product folder (`<name>/` with `<name>.nc` '
            'and `<name>_anc.nc`) or hGRS L2A NetCDF file (EnMAP, PRISMA)')


def open_rrs(path):
    """
    Open the remote-sensing reflectance of an L2A product for :class:`~grstbx.visual.ViewSpectral`.

    :param path: GRS Zarr store, GRS NetCDF product folder or hGRS NetCDF file
    :return: (data, title): path of the Zarr store (its pyramid levels are read directly) or lazy
             ``Rrs`` DataArray, and Markdown title of the viewer
    """
    from .subset import open_l2a

    path = str(path).rstrip('/')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    name = os.path.basename(path)
    if is_zarr(path):
        return path, f'## {name}'

    product, _ = open_l2a(path)
    rrs = product['Rrs']
    if 'time' not in rrs.coords:
        date = product.attrs.get('acquisition_date', product.attrs.get('start_date'))
        if date is None:
            raise ValueError(f'no acquisition time found in {path}')
        rrs = rrs.assign_coords(time=pd.Timestamp(date).tz_localize(None).to_datetime64())
    sensor = 'hGRS L2A' if is_hgrs(path) else 'GRS L2A'
    return rrs, f'## {name}\n{sensor}'


def points_spectra(viewer):
    """
    Full-resolution spectra of the points of the map, for all the dates of the viewer.

    :return: pandas.DataFrame, one row per point and date, one column per wavelength
    """
    points = viewer.get_points()
    native = points.to_crs(viewer.data.crs)
    rows = []
    for ipoint, (point, xy) in enumerate(zip(points.geometry, native.geometry)):
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            spectrum = viewer.data.pixel(idate, xy.x, xy.y)
            if spectrum is None:
                continue
            row = {'point': ipoint, 'lon': point.x, 'lat': point.y, 'time': date}
            row.update({f'Rrs_{wl:g}': float(v) for wl, v in zip(spectrum.wl.values, spectrum.values)})
            rows.append(row)
    return pd.DataFrame(rows)


def polygons_statistics(viewer):
    """
    Statistics of the full-resolution Rrs of the valid pixels within each polygon of the map.

    :return: pandas.DataFrame, one row per polygon, date and statistic (median, mean, std, q05, q25, q75,
             q95, count), one column per wavelength
    """
    from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster

    n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
    rows = []
    for ipoly in range(n_poly):
        aoi = viewer.get_geom(crs=viewer.data.crs, index=ipoly)
        xmin, ymin, xmax, ymax = aoi.total_bounds
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            rrs = viewer.data.layer(idate, list(viewer.wls), level=0)
            try:
                rrs = rrs.rio.clip_box(xmin, ymin, xmax, ymax).rio.clip(aoi.geometry.values, drop=True)
            except (NoDataInBounds, OneDimensionalRaster):   # polygon outside the image, or too small
                continue
            stacked = rrs.stack(pixel=['y', 'x']).dropna('pixel', how='all')
            if stacked.sizes['pixel'] == 0:
                continue
            stats = {'median': stacked.median('pixel'), 'mean': stacked.mean('pixel'),
                     'std': stacked.std('pixel'), 'count': stacked.count('pixel')}
            for q in (0.05, 0.25, 0.75, 0.95):
                stats[f'q{q * 100:02.0f}'] = stacked.quantile(q, dim='pixel').drop_vars('quantile')
            for stat, values in stats.items():
                row = {'polygon': ipoly, 'time': date, 'statistic': stat}
                row.update({f'Rrs_{wl:g}': float(v) for wl, v in zip(values.wl.values, values.values)})
                rows.append(row)
    return pd.DataFrame(rows)


def _csv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def _geojson(gdf):
    return io.StringIO(gdf.to_json())


def _exports(viewer):
    """Download buttons of the points, polygons, spectra and statistics drawn on the map."""

    def polygons():
        n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
        import geopandas as gpd
        if n_poly == 0:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        return pd.concat([viewer.get_geom(index=i) for i in range(n_poly)], ignore_index=True)

    buttons = [
        pn.widgets.FileDownload(callback=lambda: _csv(points_spectra(viewer)), filename='points_spectra.csv',
                                label='Spectra of the points (CSV)', button_type='primary'),
        pn.widgets.FileDownload(callback=lambda: _csv(polygons_statistics(viewer)),
                                filename='polygons_statistics.csv',
                                label='Statistics of the polygons (CSV)', button_type='primary'),
        pn.widgets.FileDownload(callback=lambda: _geojson(viewer.get_points()), filename='points.geojson',
                                label='Points (GeoJSON)'),
        pn.widgets.FileDownload(callback=lambda: _geojson(polygons()), filename='polygons.geojson',
                                label='Polygons (GeoJSON)'),
    ]
    return pn.Column(
        '### Export',
        'Full-resolution data of the points and polygons drawn on the map, for all the dates. '
        'The polygon statistics read all the pixels of the polygons: large polygons take a while.',
        pn.Row(*buttons))


def is_product(path):
    """True if ``path`` is an L2A product opened by :func:`open_rrs`: Zarr store, NetCDF file, or GRS NetCDF
    product folder ``<name>/`` holding ``<name>.nc``."""
    path = str(path).rstrip('/')
    name = os.path.basename(path)
    return is_zarr(path) or (os.path.isfile(path) and path.endswith('.nc')) or (
        os.path.isdir(path) and os.path.isfile(os.path.join(path, name + '.nc')))


def _file_browser(root, path_input):
    """
    Compact file browser of the server for the sidebar: a click on a folder opens it, a click on a
    product (see :func:`is_product`) writes its path into ``path_input``.
    """
    folder = pn.widgets.TextInput(name='Folder', value=str(root))
    entries = pn.widgets.MultiSelect(options={}, size=16, sizing_mode='stretch_width')
    parent = pn.widgets.Button(name='Parent folder', icon='arrow-up', button_type='light')

    def refresh(*events):
        try:
            names = sorted(n for n in os.listdir(folder.value) if not n.startswith('.'))
        except OSError as exc:
            pn.state.notifications.error(str(exc))
            return
        paths = [os.path.join(folder.value, n) for n in names]
        products = [p for p in paths if is_product(p)]
        folders = [p for p in paths if os.path.isdir(p) and p not in products]
        options = {'\N{FILE FOLDER} ' + os.path.basename(p): p for p in folders}
        options.update({'\N{GLOBE WITH MERIDIANS} ' + os.path.basename(p): p for p in products})
        entries.options = options

    def on_click(event):
        if not event.new:
            return
        selected = event.new[0]
        if is_product(selected):
            path_input.value = selected
        else:
            entries.value = []
            folder.value = selected

    def on_parent(event):
        folder.value = os.path.dirname(folder.value.rstrip('/')) or '/'

    folder.param.watch(refresh, 'value')
    entries.param.watch(on_click, 'value')
    parent.on_click(on_parent)
    refresh()
    return pn.Column(folder, parent, entries,
                     pn.pane.Markdown('\N{GLOBE WITH MERIDIANS}: L2A product, \N{FILE FOLDER}: folder'),
                     sizing_mode='stretch_width')


def create_app(path=None, root=None, reproject=True, basemap=None, cmap=None, width=1000, height=700):
    """
    Build the dashboard (one per browser session).

    :param path: product opened at start-up (optional)
    :param root: directory of the file browser of the sidebar (default: directory of ``path`` or current
                 directory)
    :param reproject: display over basemaps (web Mercator)
    :param basemap: initial basemap (see :data:`grstbx.visual.BASEMAPS`)
    :param cmap: initial colormap of the single bands (see :class:`~grstbx.visual.ViewSpectral`)
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :return: panel template
    """
    from . import visual

    pn.extension(notifications=True)
    if root is None:
        root = os.path.dirname(str(path).rstrip('/')) if path else os.getcwd()

    path_input = pn.widgets.TextInput(name='Product', value=str(path or ''),
                                      placeholder='/path/to/product.zarr')
    browser = _file_browser(root, path_input)
    reproject_box = pn.widgets.Checkbox(name='Basemap (reproject to web Mercator)', value=reproject)
    basemap_select = pn.widgets.Select(name='Basemap', options=list(visual.BASEMAPS),
                                       value=basemap or visual.DEFAULT_BASEMAP)
    cmap_select = pn.widgets.Select(name='Colormap of the bands', options=list(visual.RRS_COLORMAPS),
                                    value=next((label for label, name in visual.RRS_COLORMAPS.items()
                                                if cmap in (label, name)), list(visual.RRS_COLORMAPS)[0]))
    range_input = pn.widgets.EditableRangeSlider(name='Initial Rrs range (sr-1)', start=0, end=0.1,
                                                 value=(0, 0.02), step=0.001)
    load_button = pn.widgets.Button(name='Open', button_type='primary')

    main = pn.Column(sizing_mode='stretch_width')
    intro = pn.pane.Markdown(
        '## Open an L2A product\n\n'
        f'Enter the path of a {ACCEPTED} on the server, or select it in the file browser, then click **Open**.\n\n'
        'Then:\n\n'
        '- pan and zoom the map: only the pixels displayed are read, from the Zarr pyramid levels;\n'
        '- hover the map to plot the full-resolution spectrum under the mouse (dashed line);\n'
        '- add points with the **Point Draw** tool (click) to plot their spectra;\n'
        '- draw polygons with the **Polygon Draw** tool (double-click to start and to close);\n'
        '- download the spectra of the points and the statistics of the polygons below the map.')
    main.append(intro)

    def on_load(event):
        product = path_input.value.strip()
        if not product:
            pn.state.notifications.warning('Enter or select a product first.')
            return
        load_button.loading = True
        main.loading = True
        try:
            data, title = open_rrs(product)
            viewer = visual.ViewSpectral(
                data, reproject=reproject_box.value, basemap=basemap_select.value, cmap=cmap_select.value,
                minmaxvalues=tuple(range_input.value), minmax=(0, max(0.1, range_input.value[1])),
                width=width, height=height, title=title)
            main.objects = [viewer.visu(), _exports(viewer)]
        except Exception as exc:
            pn.state.notifications.error(f'Cannot open {product}: {exc}', duration=0)
            main.objects = [intro]
        finally:
            load_button.loading = False
            main.loading = False

    load_button.on_click(on_load)
    if path:
        pn.state.onload(lambda: on_load(None))

    sidebar = [
        path_input,
        pn.Card(browser, title='File browser', collapsed=bool(path), sizing_mode='stretch_width'),
        pn.Card(reproject_box, basemap_select, cmap_select, range_input, title='Display', collapsed=True,
                sizing_mode='stretch_width'),
        load_button,
    ]
    return pn.template.FastListTemplate(title='grstbx L2A viewer', sidebar=sidebar, main=[main],
                                        sidebar_width=380, header_background='#1f4e79')


def serve(path=None, root=None, port=5006, address=None, show=True, allow_websocket_origin=None, **kwargs):
    """
    Serve the dashboard (blocking).

    :param path: product opened at start-up (optional)
    :param root: directory of the file browser
    :param port: port of the server
    :param address: address to listen on (e.g. '0.0.0.0' to accept connections from other computers)
    :param show: open the dashboard in the web browser
    :param allow_websocket_origin: list of hosts allowed to connect (e.g. ['myserver:5006'])
    :param kwargs: other arguments of :func:`create_app`
    """
    pn.serve(lambda: create_app(path=path, root=root, **kwargs), port=port, address=address, show=show,
             websocket_origin=allow_websocket_origin, title='grstbx L2A viewer')


def main(argv=None):
    """Command line entry point: ``grstbx-viewer``."""
    parser = argparse.ArgumentParser(prog='grstbx-viewer',
                                     description='Web dashboard to explore a GRS or hGRS L2A product.')
    parser.add_argument('path', nargs='?', help=f'product opened at start-up: {ACCEPTED}')
    parser.add_argument('--root', help='directory of the file browser (default: directory of the product, '
                                       'or current directory)')
    parser.add_argument('--port', type=int, default=5006, help='port of the server (default: 5006)')
    parser.add_argument('--address', help="address to listen on, e.g. 0.0.0.0 to accept connections from other "
                                          "computers (default: localhost)")
    parser.add_argument('--allow-websocket-origin', action='append',
                        help='host[:port] allowed to connect, repeat for several hosts (needed with --address)')
    parser.add_argument('--no-show', action='store_true', help='do not open the web browser')
    parser.add_argument('--no-reproject', action='store_true',
                        help='display in the projection of the image, without basemap')
    parser.add_argument('--basemap', help='initial basemap, e.g. "CARTO Positron"')
    parser.add_argument('--cmap', help="initial colormap of the bands, e.g. 'cmo.turbid'")
    parser.add_argument('--stadia-api-key', help='API key of the Stadia Maps basemaps (or STADIA_API_KEY)')
    args = parser.parse_args(argv)

    if args.stadia_api_key:
        from .visual import set_stadia_api_key
        set_stadia_api_key(args.stadia_api_key)
    serve(path=args.path, root=args.root, port=args.port, address=args.address, show=not args.no_show,
          allow_websocket_origin=args.allow_websocket_origin, reproject=not args.no_reproject,
          basemap=args.basemap, cmap=args.cmap)


if __name__ == '__main__':
    main()
