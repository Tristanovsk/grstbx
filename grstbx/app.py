"""
Web dashboard to explore GRS (or hGRS) products: L2A images with :class:`~grstbx.visual.ViewSpectral`,
L2B images with :class:`~grstbx.visual.ViewParam`, one product, a time series of products or a
multi-temporal datacube.

The dashboard is a `Panel <https://panel.holoviz.org>`_ application served by a Python process, which
reads the pixels displayed on the screen and the spectra or time series under the mouse: start it with

.. code-block:: console

    grstbx-viewer                                   # empty dashboard, choose the products in the sidebar
    grstbx-viewer /data/S2A_MSIL2AGRS_..._V3.0.1.zarr --root /data/satellite
    grstbx-viewer "/data/L2B/BEFR/2024/*/*/*.nc"    # time series of L2B products (glob pattern)
    grstbx-viewer --address 0.0.0.0 --port 5006 --allow-websocket-origin myserver:5006

or from Python with :func:`serve`. The products are read on the machine running the server: with
``--address 0.0.0.0``, the dashboard is reachable from other computers of the network.
"""

import argparse
import glob
import io
import os
import re
import warnings

import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from .driver import is_hgrs, is_zarr

ACCEPTED = ('GRS L2A or L2B Zarr stores (`*.zarr`), GRS L2A NetCDF product folders (`<name>/` with '
            '`<name>.nc` and `<name>_anc.nc`), GRS L2B NetCDF files, hGRS L2A NetCDF files (EnMAP, PRISMA), '
            'or a NetCDF datacube with a `time` dimension')

# variables of the L2B products that are not parameters to display
NOT_PARAMETERS = ('flags', 'mask', 'spatial_ref', 'dem', 'valid_pix_prop')
# order of the L2B parameters in the viewer (by prefix), the other ones come last
PARAMETER_ORDER = ('Chla', 'SPM', 'TURB', 'acdom', 'Kd')
# datacubes smaller than this (bytes) are loaded in memory: fast time series under the mouse and exports
MAX_IN_MEMORY = 1e9


# -- opening of the products ------------------------------------------------------------------------

def is_product(path):
    """True if ``path`` can be opened by :func:`open_products`: Zarr store, NetCDF file, or GRS NetCDF
    product folder ``<name>/`` holding ``<name>.nc``."""
    path = str(path).rstrip('/')
    name = os.path.basename(path)
    return is_zarr(path) or (os.path.isfile(path) and path.endswith('.nc')) or (
        os.path.isdir(path) and os.path.isfile(os.path.join(path, name + '.nc')))


def expand_paths(paths):
    """
    List of products from paths and glob patterns.

    :param paths: path, glob pattern (e.g. ``'/data/L2B/BEFR/2024/*/*/*.nc'``), or list of them, or text
                  with one per line
    :return: list of existing paths (patterns expanded and sorted), without duplicates
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = str(paths).splitlines()
    out = []
    for item in paths:
        item = str(item).strip().rstrip('/')
        if not item:
            continue
        matches = sorted(glob.glob(item)) if glob.has_magic(item) else [item]
        if not matches:
            raise FileNotFoundError(f'no product matches {item}')
        for match in matches:
            if not os.path.exists(match):
                raise FileNotFoundError(match)
            if match not in out:
                out.append(match)
    return out


def _image_time(ds, path):
    """Acquisition time of an image: ``time`` coordinate, attributes or date in the product name."""
    if 'time' in ds.coords and ds['time'].size == 1:
        return np.datetime64(pd.Timestamp(ds['time'].values.ravel()[0]).tz_localize(None), 'ns')
    for attr in ('acquisition_date', 'start_date', 'PRODUCT_START_TIME'):
        if attr in ds.attrs:
            return np.datetime64(pd.Timestamp(ds.attrs[attr]).tz_localize(None), 'ns')
    match = re.search(r'(\d{8}T\d{6})', os.path.basename(path))
    if match:
        return np.datetime64(pd.Timestamp(match.group(1)), 'ns')
    raise ValueError(f'no acquisition time found in {path}')


def _open_levels(path):
    """Resolution levels (full resolution first) of a product, as lazy xarray.Datasets."""
    from .driver import open_zarr_image, zarr_levels, open_hgrs_l2a
    from .subset import open_l2a

    if is_zarr(path):
        return [open_zarr_image(path, level=level) for level in range(max(len(zarr_levels(path)), 1))]
    if is_hgrs(path):
        return [open_hgrs_l2a(path)[0]]
    if os.path.isdir(path):
        return [open_l2a(path)[0]]
    return [xr.open_dataset(path, decode_coords='all', chunks={})]


def _is_l2a(ds):
    return 'Rrs' in ds and 'wl' in ds['Rrs'].dims


def _parameters(ds, dims=('y', 'x')):
    """Variables of an L2B product (or datacube) displayed by ViewParam: float maps."""
    names = [name for name, var in ds.data_vars.items()
             if set(var.dims) == set(dims) and var.dtype.kind == 'f' and name not in NOT_PARAMETERS]

    def rank(name):
        return next((i for i, prefix in enumerate(PARAMETER_ORDER) if name.startswith(prefix)), len(PARAMETER_ORDER))

    return sorted(names, key=rank)


def _layers(ds, kind, params=None, mask_invalid=True):
    """Layers displayed by the viewers: Rrs (L2A) or the parameters (L2B), masked with ``mask != 0``."""
    if kind == 'L2A':
        return ds['Rrs']
    layers = ds[params]
    if mask_invalid and 'mask' in ds:
        layers = layers.where(ds['mask'] == 0)
    return layers


def open_products(paths, mask_invalid=True, min_size=512):
    """
    Open products for the viewers, lazily.

    The products can be on different grids (e.g. subsets of several tiles), but must share the same CRS.

    :param paths: path, glob pattern, or list of them (see :func:`expand_paths`): L2A or L2B products
                  (one image each, all of the same level), or a single NetCDF datacube with a ``time``
                  dimension
    :param mask_invalid: L2B: hide the pixels with ``mask != 0`` (not valid water pixels)
    :param min_size: size (pixels) of the coarsest level computed for the products without pyramid
    :return: (data, kind, params, title): :class:`~grstbx.visual.Multiscale`, ``'L2A'`` or ``'L2B'``,
             displayed parameters (L2B, None for L2A), Markdown title of the viewer
    """
    from .visual import Multiscale

    paths = expand_paths(paths)
    if not paths:
        raise ValueError('no product given')

    # NetCDF datacube: one file with a time dimension
    if len(paths) == 1 and not is_zarr(paths[0]) and os.path.isfile(paths[0]) and not is_hgrs(paths[0]):
        cube = xr.open_dataset(paths[0], decode_coords='all', chunks={})
        if 'time' in cube.dims and cube.sizes['time'] > 1:
            cube = cube.sortby('time')
            kind = 'L2A' if _is_l2a(cube) else 'L2B'
            params = None if kind == 'L2A' else _parameters(cube, dims=('time', 'y', 'x'))
            if kind == 'L2B' and not params:
                raise ValueError(f'no parameter to display in {paths[0]}')
            layers = _layers(cube, kind, params, mask_invalid)
            if layers.nbytes < MAX_IN_MEMORY:
                layers = layers.load()
            data = Multiscale.from_xarray(layers, min_size=min_size)
            data.scheduler = 'synchronous'   # dask scheduler of the exports (NetCDF)
            title = f'## {os.path.basename(paths[0])}\n{kind} datacube, {cube.sizes["time"]} dates'
            return data, kind, params, title

    products = [(path, _open_levels(path)) for path in paths]
    kinds = {'L2A' if _is_l2a(levels[0]) else 'L2B' for _, levels in products}
    if len(kinds) > 1:
        raise ValueError('L2A and L2B products cannot be displayed together')
    kind = kinds.pop()

    params = None
    if kind == 'L2B':
        # parameters common to all the products
        params = _parameters(products[0][1][0])
        for _, levels in products[1:]:
            names = set(_parameters(levels[0]))
            params = [param for param in params if param in names]
        if not params:
            raise ValueError('no parameter common to all the products')

    crs = {levels[0].rio.crs for _, levels in products}
    if len(crs) > 1:
        raise ValueError(f'the products are in different CRS ({", ".join(map(str, crs))}): open the products '
                         'of one UTM zone at a time')

    times = [_image_time(levels[0], path) for path, levels in products]
    images = [[_layers(level, kind, params, mask_invalid) for level in levels] for _, levels in products]
    order = np.argsort(times)
    data = Multiscale([images[i] for i in order], np.array(times)[order], min_size=min_size)
    data.scheduler = None if all(is_zarr(path) for path in paths) else 'synchronous'

    if len(paths) == 1:
        level = 'hGRS L2A' if is_hgrs(paths[0]) else f'GRS {kind}'
        title = f'## {os.path.basename(paths[0])}\n{level}'
    else:
        dates = pd.DatetimeIndex(data.times)
        title = f'## {len(paths)} {kind} products\nfrom {dates[0].date()} to {dates[-1].date()}'
    return data, kind, params, title


# -- exports ------------------------------------------------------------------------------------------

def _is_spectral(viewer):
    return hasattr(viewer, 'wls')


def _as_layers(obj, viewer):
    """(labels, DataArray with a 'layer' first dimension) of an image or pixel of the viewer."""
    if _is_spectral(viewer):
        da = obj.sel(wl=list(viewer.wls)).rename(wl='layer')
        return [f'Rrs_{wl:g}' for wl in viewer.wls], da
    return list(viewer.params), obj[list(viewer.params)].to_array('layer')


def _compute(objects, viewer):
    """
    Compute lazy xarray objects together, in a single dask graph. NetCDF (HDF5) reads are serialized by a
    lock: they are faster on a single thread (see :class:`~grstbx.driver.L2grs`).
    """
    import dask

    with warnings.catch_warnings():   # all-NaN pixels or polygons
        warnings.simplefilter('ignore', RuntimeWarning)
        return dask.compute(*objects, scheduler=getattr(viewer.data, 'scheduler', None))


def points_values(viewer):
    """
    Full-resolution values of the points of the map, for all the dates of the viewer: spectra
    (:class:`~grstbx.visual.ViewSpectral`) or parameters (:class:`~grstbx.visual.ViewParam`).

    :return: pandas.DataFrame, one row per point and date, one column per wavelength or parameter
    """
    points = viewer.get_points()
    native = points.to_crs(viewer.data.crs)
    lazy, rows = [], []
    for ipoint, (point, xy) in enumerate(zip(points.geometry, native.geometry)):
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            xmin, ymin, xmax, ymax = viewer.data.bounds(idate)
            if not (xmin <= xy.x <= xmax and ymin <= xy.y <= ymax):   # point outside the image
                continue
            pixel = viewer.data.images[idate][0].sel(x=xy.x, y=xy.y, method='nearest')
            labels, layers = _as_layers(pixel, viewer)
            lazy.append(layers)
            rows.append({'point': ipoint, 'lon': point.x, 'lat': point.y, 'time': date})
    for row, values in zip(rows, _compute(lazy, viewer)):
        row.update({label: float(v) for label, v in zip(labels, values.values)})
    return pd.DataFrame(rows)



def polygons_statistics(viewer):
    """
    Statistics of the full-resolution valid pixels within each polygon of the map, for all the dates of
    the viewer: Rrs (:class:`~grstbx.visual.ViewSpectral`) or parameters (:class:`~grstbx.visual.ViewParam`).

    :return: pandas.DataFrame, one row per polygon, date and statistic (median, mean, std, q05, q25, q75,
             q95, count), one column per wavelength or parameter
    """
    from rasterio.features import geometry_mask

    n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
    lazy, keys, masks = [], [], {}
    for ipoly in range(n_poly):
        aoi = viewer.get_geom(crs=viewer.data.crs, index=ipoly)
        xmin, ymin, xmax, ymax = aoi.total_bounds
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            image = viewer.data.images[idate][0]
            ys = image.y.values
            # bounding box of the polygon (lazy slicing)
            image = image.sel(x=slice(xmin, xmax), y=slice(ymax, ymin) if ys[0] > ys[-1] else slice(ymin, ymax))
            if image.sizes['x'] == 0 or image.sizes['y'] == 0:   # polygon outside the image
                continue
            # pixels inside the polygon; the mask is computed once per grid (e.g. for all the dates of a
            # datacube)
            grid = (ipoly, image.sizes['y'], image.sizes['x'], tuple(image.rio.transform()))
            if grid not in masks:
                masks[grid] = xr.DataArray(
                    geometry_mask(aoi.geometry.values, out_shape=grid[1:3], transform=image.rio.transform(),
                                  invert=True), dims=('y', 'x'), coords={'y': image.y, 'x': image.x})
            labels, layers = _as_layers(image.where(masks[grid]), viewer)
            lazy.append(layers)
            keys.append((ipoly, date))

    rows = []
    for (ipoly, date), layers in zip(keys, _compute(lazy, viewer)):
        values = layers.values.reshape(layers.shape[0], -1)
        values = values[:, np.isfinite(values).any(axis=0)]
        if values.shape[1] == 0:   # no valid pixel
            continue
        with warnings.catch_warnings():   # all-NaN layers
            warnings.simplefilter('ignore', RuntimeWarning)
            stats = {'median': np.nanmedian(values, axis=1), 'mean': np.nanmean(values, axis=1),
                     'std': np.nanstd(values, axis=1), 'count': np.isfinite(values).sum(axis=1)}
            for q in (0.05, 0.25, 0.75, 0.95):
                stats[f'q{q * 100:02.0f}'] = np.nanquantile(values, q, axis=1)
        for stat, column in stats.items():
            row = {'polygon': ipoly, 'time': date, 'statistic': stat}
            row.update({label: float(v) for label, v in zip(labels, column)})
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
    """Download buttons of the points, polygons, values and statistics drawn on the map."""
    import geopandas as gpd

    def polygons():
        n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
        if n_poly == 0:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        return pd.concat([viewer.get_geom(index=i) for i in range(n_poly)], ignore_index=True)

    values = 'Spectra' if _is_spectral(viewer) else 'Parameters'
    buttons = [
        pn.widgets.FileDownload(callback=lambda: _csv(points_values(viewer)), filename='points_values.csv',
                                label=f'{values} of the points (CSV)', button_type='primary'),
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
        f'Full-resolution {values.lower()} of the points and statistics of the polygons drawn on the map, for '
        f'all the dates ({len(viewer.indexes)}). The polygon statistics read all the pixels of the polygons: '
        'large polygons or long time series take a while.',
        pn.Row(*buttons))


# -- dashboard ---------------------------------------------------------------------------------------

def _file_browser(root, products_input):
    """
    Compact file browser of the server for the sidebar: a click on a folder opens it, a click on a
    product (see :func:`is_product`) adds it to ``products_input``.
    """
    folder = pn.widgets.TextInput(name='Folder', value=str(root))
    entries = pn.widgets.MultiSelect(options={}, size=16, sizing_mode='stretch_width')
    parent = pn.widgets.Button(name='Parent folder', icon='arrow-up', button_type='light')
    add_all = pn.widgets.Button(name='Add all the products', icon='list-check', button_type='light')

    def listing():
        try:
            names = sorted(n for n in os.listdir(folder.value) if not n.startswith('.'))
        except OSError as exc:
            pn.state.notifications.error(str(exc))
            return [], []
        paths = [os.path.join(folder.value, n) for n in names]
        products = [p for p in paths if is_product(p)]
        return [p for p in paths if os.path.isdir(p) and p not in products], products

    def refresh(*events):
        folders, products = listing()
        options = {'\N{FILE FOLDER} ' + os.path.basename(p): p for p in folders}
        options.update({'\N{GLOBE WITH MERIDIANS} ' + os.path.basename(p): p for p in products})
        entries.options = options

    def add(paths):
        current = [line.strip() for line in products_input.value.splitlines() if line.strip()]
        products_input.value = '\n'.join(current + [p for p in paths if p not in current])

    def on_click(event):
        if not event.new:
            return
        selected = event.new[0]
        entries.value = []
        if is_product(selected):
            add([selected])
        else:
            folder.value = selected

    def on_parent(event):
        folder.value = os.path.dirname(folder.value.rstrip('/')) or '/'

    folder.param.watch(refresh, 'value')
    entries.param.watch(on_click, 'value')
    parent.on_click(on_parent)
    add_all.on_click(lambda event: add(listing()[1]))
    refresh()
    return pn.Column(folder, pn.Row(parent, add_all), entries,
                     pn.pane.Markdown('\N{GLOBE WITH MERIDIANS}: product (click to add it), '
                                      '\N{FILE FOLDER}: folder (click to open it)'),
                     sizing_mode='stretch_width')


def create_app(path=None, root=None, reproject=True, basemap=None, cmap=None, width=1000, height=700):
    """
    Build the dashboard (one per browser session).

    :param path: product(s) opened at start-up (optional): path, glob pattern or list of them
    :param root: directory of the file browser of the sidebar (default: directory of the first product or
                 current directory)
    :param reproject: display over basemaps (web Mercator)
    :param basemap: initial basemap (see :data:`grstbx.visual.BASEMAPS`)
    :param cmap: initial colormap (see :class:`~grstbx.visual.ViewSpectral`)
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :return: panel template
    """
    from . import visual

    pn.extension(notifications=True)
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path or [])
    if root is None:
        first = str(paths[0]).rstrip('/') if paths else ''
        root = os.path.dirname(first.split('*')[0]) if first else os.getcwd()
        if not os.path.isdir(root):
            root = os.getcwd()

    products_input = pn.widgets.TextAreaInput(
        name='Products', value='\n'.join(map(str, paths)), height=110, sizing_mode='stretch_width',
        placeholder='One path or glob pattern per line, e.g.\n/data/L2A/product.zarr\n/data/L2B/BEFR/2024/*/*/*.nc')
    browser = _file_browser(root, products_input)
    clear_button = pn.widgets.Button(name='Clear', button_type='light')
    reproject_box = pn.widgets.Checkbox(name='Basemap (reproject to web Mercator)', value=reproject)
    basemap_select = pn.widgets.Select(name='Basemap', options=list(visual.BASEMAPS),
                                       value=basemap or visual.DEFAULT_BASEMAP)
    cmap_select = pn.widgets.Select(name='Colormap', options=list(visual.RRS_COLORMAPS),
                                    value=next((label for label, name in visual.RRS_COLORMAPS.items()
                                                if cmap in (label, name)), list(visual.RRS_COLORMAPS)[0]))
    rrs_range = pn.widgets.EditableRangeSlider(name='L2A: initial Rrs range (sr-1)', start=0, end=0.1,
                                               value=(0, 0.02), step=0.001)
    param_range = pn.widgets.EditableRangeSlider(name='L2B: initial range of the parameters', start=0, end=100,
                                                 value=(0, 10), step=0.1)
    mask_box = pn.widgets.Checkbox(name='L2B: hide the invalid pixels (mask != 0)', value=True)
    load_button = pn.widgets.Button(name='Open', button_type='primary')

    main = pn.Column(sizing_mode='stretch_width')
    intro = pn.pane.Markdown(
        '## Open GRS products\n\n'
        f'Enter the paths of {ACCEPTED} on the server, one path or glob pattern per line, or add them with '
        'the file browser, then click **Open**. Several products make a time series: they can be on different '
        'grids, but must be in the same CRS (UTM zone).\n\n'
        '- **L2A** products are displayed with the spectral viewer: pick the band or the true-colour '
        'composite, hover the map to plot the full-resolution spectrum under the mouse (dashed line), '
        'add points with the **Point Draw** tool to plot their spectra;\n'
        '- **L2B** products are displayed with the parameter viewer: pick the parameter and the date, hover '
        'the map to plot the time series under the mouse, add points to plot their time series.\n\n'
        'Pan and zoom the map: only the pixels displayed are read, from the Zarr pyramid levels. Draw '
        'polygons with the **Polygon Draw** tool (double-click to start and to close), and download the '
        'values of the points and the statistics of the polygons below the map.')
    main.append(intro)

    def on_load(event):
        if not products_input.value.strip():
            pn.state.notifications.warning('Enter or add products first.')
            return
        load_button.loading = True
        main.loading = True
        try:
            data, kind, params, title = open_products(products_input.value, mask_invalid=mask_box.value)
            options = dict(reproject=reproject_box.value, basemap=basemap_select.value, cmap=cmap_select.value,
                           width=width, height=height, title=title)
            if kind == 'L2A':
                viewer = visual.ViewSpectral(data, minmaxvalues=tuple(rrs_range.value),
                                             minmax=(0, max(0.1, rrs_range.value[1])), **options)
            else:
                viewer = visual.ViewParam(data, params=params, minmaxvalues=tuple(param_range.value),
                                          minmax=(0, max(100, param_range.value[1])), **options)
            main.objects = [viewer.visu(), _exports(viewer)]
        except Exception as exc:
            pn.state.notifications.error(f'Cannot open the products: {exc}', duration=0)
            main.objects = [intro]
        finally:
            load_button.loading = False
            main.loading = False

    load_button.on_click(on_load)
    clear_button.on_click(lambda event: setattr(products_input, 'value', ''))
    if paths:
        pn.state.onload(lambda: on_load(None))

    sidebar = [
        products_input,
        pn.Row(load_button, clear_button),
        pn.Card(browser, title='File browser', collapsed=bool(paths), sizing_mode='stretch_width'),
        pn.Card(reproject_box, basemap_select, cmap_select, rrs_range, param_range, mask_box, title='Display',
                collapsed=True, sizing_mode='stretch_width'),
    ]
    return pn.template.FastListTemplate(title='grstbx viewer', sidebar=sidebar, main=[main],
                                        sidebar_width=380, header_background='#1f4e79')


def serve(path=None, root=None, port=5006, address=None, show=True, allow_websocket_origin=None, **kwargs):
    """
    Serve the dashboard (blocking).

    :param path: product(s) opened at start-up (optional): path, glob pattern or list of them
    :param root: directory of the file browser
    :param port: port of the server
    :param address: address to listen on (e.g. '0.0.0.0' to accept connections from other computers)
    :param show: open the dashboard in the web browser
    :param allow_websocket_origin: list of hosts allowed to connect (e.g. ['myserver:5006']), added to the
                                   names of the local machine (localhost, 127.0.0.1, host name, address)
    :param kwargs: other arguments of :func:`create_app`
    """
    pn.serve(lambda: create_app(path=path, root=root, **kwargs), port=port, address=address, show=show,
             websocket_origin=_websocket_origins(port, address, allow_websocket_origin),
             title='grstbx viewer')


def _websocket_origins(port, address=None, extra=None):
    """
    Hosts allowed to open the websocket of the dashboard. Bokeh allows only ``localhost:<port>`` by default,
    so that a page opened as ``127.0.0.1:<port>`` (or with the host name) stays empty.
    """
    import socket

    hosts = {'localhost', '127.0.0.1', socket.gethostname(), socket.getfqdn()}
    if address and address not in ('0.0.0.0', '::'):
        hosts.add(address)
    origins = sorted(f'{host}:{port}' for host in hosts if host)
    return origins + [origin for origin in (extra or []) if origin not in origins]


def main(argv=None):
    """Command line entry point: ``grstbx-viewer``."""
    parser = argparse.ArgumentParser(prog='grstbx-viewer',
                                     description='Web dashboard to explore GRS L2A and L2B products, '
                                                 'single images or time series.')
    parser.add_argument('path', nargs='*',
                        help=f'products opened at start-up (paths or quoted glob patterns): {ACCEPTED}')
    parser.add_argument('--root', help='directory of the file browser (default: directory of the first '
                                       'product, or current directory)')
    parser.add_argument('--port', type=int, default=5006, help='port of the server (default: 5006)')
    parser.add_argument('--address', help="address to listen on, e.g. 0.0.0.0 to accept connections from other "
                                          "computers (default: localhost)")
    parser.add_argument('--allow-websocket-origin', action='append',
                        help='host[:port] allowed to connect, repeat for several hosts; localhost, 127.0.0.1 and '
                             'the host name of the machine are always allowed')
    parser.add_argument('--no-show', action='store_true', help='do not open the web browser')
    parser.add_argument('--no-reproject', action='store_true',
                        help='display in the projection of the images, without basemap')
    parser.add_argument('--basemap', help='initial basemap, e.g. "CARTO Positron"')
    parser.add_argument('--cmap', help="initial colormap, e.g. 'cmo.turbid'")
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
