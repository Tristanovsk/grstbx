import json
import os
import shutil

import numpy as np
import pandas as pd
import panel as pn
import pytest

from grstbx.app import (_product_file, catalogue, find_products, is_product, is_stac_item, open_products, product_info,
                        select_products, SeveralZones)


def test_product_info():
    info = product_info('/data/S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0.nc')
    assert info['level'] == 'L2B'
    assert info['tile'] == '31TGJ'
    assert info['satellite'] == 'S2A'
    assert info['time'] == pd.Timestamp('2025-01-03 10:33:39')
    assert info['processing_time'] == pd.Timestamp('2025-01-03 15:00:00')
    assert info['version'] == '3.1.0'


def test_find_products(l2b_files, tmp_path):
    products = find_products(tmp_path)
    assert list(products.path) == l2b_files
    assert products.time.is_monotonic_increasing


def test_select_products_last_processing():
    names = ['S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0',
             'S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.2.0dev',
             'S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250101T150000_V3.0.0',
             'S2A_MSIL2B_20250108T103421_N0511_R108_T31TGJ_20250108T150000_V3.1.0',
             'S2B_MSIL2B_20250110T103421_N0511_R108_T32TLP_20250110T150000_V3.1.0']
    products = pd.DataFrame([product_info(name + '.nc') for name in names])
    selected = select_products(products, tile='31TGJ', start='2025-01-01', end='2025-01-08')
    # one product per acquisition: the latest processing, the release preferred to a dev version
    # processed at the same time
    assert list(selected.name) == [names[0], names[3]]
    assert len(select_products(products, last_processing=False)) == 5


def test_l2b_product_folders(l2b_files, tmp_path):
    """GRSl2bgen --product_dir layout: <name>/<name>.nc with its log and STAC Item."""
    root = tmp_path / 'product_dir'
    expected = []
    for path in l2b_files:
        name = os.path.basename(path)[:-3]
        folder = root / os.path.relpath(os.path.dirname(path), tmp_path) / name
        folder.mkdir(parents=True)
        shutil.copy(path, folder / f'{name}.nc')
        (folder / f'{name}.log').write_text('log')
        (folder / f'{name}.json').write_text('{}')
        expected.append(str(folder / f'{name}.nc'))
    folder = os.path.dirname(expected[0])
    assert is_product(folder) and _product_file(folder) == expected[0]
    products = find_products(root)
    assert list(products.path) == expected
    assert list(products.level) == ['L2B'] * 3
    assert list(catalogue([folder]).path) == expected[:1]


def test_grs2_l2a_folder_unchanged(tmp_path):
    """A GRS 2 L2A NetCDF folder (with <name>_anc.nc) is opened as a folder."""
    name = 'S2A_MSIL2Agrs_20250103T103339_N0511_R108_T31TGJ_20250103T150000'
    folder = tmp_path / name
    folder.mkdir()
    (folder / f'{name}.nc').write_text('')
    (folder / f'{name}_anc.nc').write_text('')
    assert is_product(folder) and _product_file(folder) == str(folder)


def _write_item(path, data_href, datetime, cloud_cover, tile='31TGJ'):
    """Minimal STAC Item as written by GRSl2bgen --stac."""
    item = {'type': 'Feature', 'stac_version': '1.0.0', 'id': os.path.basename(path)[:-5],
            'properties': {'datetime': datetime, 'platform': 'sentinel-2b', 'processing:level': 'L2B',
                           'processing:software': {'GRSl2bgen': '1.1.0', 'GRS': '3.1.0'},
                           'eo:cloud_cover': cloud_cover, 'grid:code': f'MGRS-{tile}', 'proj:code': 'EPSG:32631'},
            'geometry': None, 'links': [],
            'assets': {'data': {'href': data_href, 'roles': ['data']},
                       'log': {'href': './x.log', 'roles': ['metadata']}}}
    with open(path, 'w') as f:
        json.dump(item, f)


@pytest.fixture
def l2b_items(l2b_files, tmp_path):
    """The L2B files in GRSl2bgen product folders <name>/ with their STAC Items (cloud cover 5, 30, 60 %)."""
    root = tmp_path / 'stac'
    items = []
    for path, cloud in zip(l2b_files, (5., 30., 60.)):
        name = os.path.basename(path)[:-3]
        folder = root / os.path.relpath(os.path.dirname(path), tmp_path) / name
        folder.mkdir(parents=True)
        shutil.copy(path, folder / f'{name}.nc')
        # acquisition time of the Item 1 s after the one of the name: the Item is the reference
        stamp = pd.Timestamp(name.split('_')[2]) + pd.Timedelta('1s')
        _write_item(folder / f'{name}.json', f'./{name}.nc', f'{stamp:%Y-%m-%dT%H:%M:%S}Z', cloud)
        items.append(str(folder / f'{name}.json'))
    return root, items


def test_products_described_by_their_stac_items(l2b_items):
    root, items = l2b_items
    products = find_products(root)
    assert list(products.item) == items
    assert list(products.path) == [item[:-5] + '.nc' for item in items]
    assert list(products.cloud_cover) == [5., 30., 60.]
    assert set(products.crs) == {'EPSG:32631'} and set(products.satellite) == {'S2B'}
    assert products.time[0] == pd.Timestamp('2025-01-03 10:33:40')


def test_catalogue_of_stac_items_glob(l2b_items):
    root, items = l2b_items
    products = catalogue(str(root / '*' / '*' / '*' / '*' / '*' / '*.json'))
    assert list(products.item) == items
    assert all(is_stac_item(item) for item in items)
    assert not is_stac_item(items[0].replace('.json', '.nc'))


def test_select_products_max_cloud_cover(l2b_items, l2b_files):
    root, items = l2b_items
    products = find_products(root)
    assert len(select_products(products, max_cloud_cover=30)) == 2
    # products without Item (no cloud cover) are kept
    both = pd.concat([products, find_products(os.path.dirname(os.path.dirname(os.path.dirname(l2b_files[0]))))])
    assert both.cloud_cover.isna().sum() == 3
    assert len(select_products(both, max_cloud_cover=10, last_processing=False)) == 4


def test_flat_layout_product_and_item_found_once(l2b_files):
    path = l2b_files[0]
    _write_item(path[:-3] + '.json', './' + os.path.basename(path), '2025-01-03T10:33:39Z', 12.)
    products = find_products(os.path.dirname(path))
    assert len(products) == 1 and products.cloud_cover[0] == 12.


def test_item_with_missing_data_asset_is_not_a_product(tmp_path):
    item = tmp_path / 'S2B_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0.json'
    _write_item(item, './missing.nc', '2025-01-03T10:33:39Z', 1.)
    assert not is_product(item) and find_products(tmp_path).empty
    (tmp_path / 'other.json').write_text('{"type": "FeatureCollection"}')
    assert not is_stac_item(tmp_path / 'other.json')


def test_search_items_local(l2b_items):
    from grstbx.app import search_items

    root, items = l2b_items
    # the synthetic Items have no bbox: an area excludes them, the other criteria select them
    assert len(search_items(str(root))) == 3
    assert len(search_items(str(root), start='2025-01-04', max_cloud_cover=50)) == 1
    assert search_items(str(root), bbox=(0, 0, 1, 1)).empty
    assert search_items(str(root), tile='32TQR').empty


def test_search_items_stac_api_paging(l2b_items):
    """STAC API Item Search: POST /search, criteria in the body, pages followed with the next links."""
    import http.server
    import threading

    from grstbx.app import search_items

    root, items = l2b_items
    features = []
    for path in items:
        with open(path) as f:
            item = json.load(f)
        item['bbox'] = [4.4, 43.4, 4.6, 43.6]
        item['assets']['data']['href'] = 'https://example.org/' + item['id'] + '.zarr'
        features.append(item)
    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            bodies.append(body)
            page = body.get('page', 0)
            links = []
            if page + 1 < len(features):
                links = [{'rel': 'next', 'href': f'http://127.0.0.1:{port}/search', 'method': 'POST',
                          'body': {'page': page + 1}, 'merge': True}]
            data = json.dumps({'type': 'FeatureCollection', 'features': [features[page]], 'links': links})
            self.send_response(200)
            self.send_header('Content-Type', 'application/geo+json')
            self.end_headers()
            self.wfile.write(data.encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        found = search_items(f'http://127.0.0.1:{port}', bbox=(4.0, 43.0, 5.0, 44.0), start='2025-01-01',
                             end='2025-01-31', collections='l2b', max_cloud_cover=50)
    finally:
        server.shutdown()
    assert len(bodies) == 3 and bodies[0]['collections'] == ['l2b'] and bodies[0]['bbox'] == [4.0, 43.0, 5.0, 44.0]
    assert bodies[0]['query'] == {'eo:cloud_cover': {'lte': 50}}
    assert bodies[0]['datetime'].startswith('2025-01-01T00:00:00') and bodies[1]['collections'] == ['l2b']
    # the third Item (60 %) is removed again on the client side
    assert list(found.cloud_cover) == [5., 30.]
    assert found.path[0] == 'https://example.org/' + os.path.basename(items[0])[:-5] + '.zarr'


@pytest.fixture
def remote_l2b(tmp_path):
    """Synthetic L2B Zarr store and its STAC Item, served over HTTP from tmp_path (like a bucket)."""
    pytest.importorskip('fsspec')
    pytest.importorskip('aiohttp')
    import functools
    import http.server
    import threading

    from conftest import l2b_product

    name = 'S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0'
    folder = tmp_path / name
    folder.mkdir()
    ds = l2b_product(pd.Timestamp('2025-01-03T10:33:39'), chla=3.0)
    ds.to_zarr(folder / f'{name}.zarr', consolidated=True)
    _write_item(folder / f'{name}.json', f'./{name}.zarr', '2025-01-03T10:33:39Z', 4.)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *args: None
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}/{name}/{name}'
    yield base, ds
    server.shutdown()


@pytest.mark.filterwarnings('ignore:Consolidated metadata:UserWarning')
def test_open_remote_zarr_and_item(remote_l2b):
    from grstbx.app import open_products

    base, ds = remote_l2b
    # remote Item: described from it (cloud cover), its relative data asset resolved to a URL
    products = catalogue(base + '.json')
    assert products.path[0] == base + '.zarr' and products.cloud_cover[0] == 4.
    for path in (base + '.json', base + '.zarr'):
        data, kind, params, title = open_products(path)
        assert kind == 'L2B' and 'Chla_OC2nasa' in params
        values = data.images[0][0]['Chla_OC2nasa'].values
        np.testing.assert_array_equal(values, ds['Chla_OC2nasa'].values)
    with pytest.raises(ValueError):
        open_products(base + '.json', max_cloud_cover=1)


def _serve_items(features):
    """Minimal STAC API (POST /search, one page) on localhost; return (server, url)."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            data = json.dumps({'type': 'FeatureCollection', 'features': features, 'links': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/geo+json')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


def test_selectfiles_search_stac_local(l2b_items):
    from grstbx import SelectFiles

    root, items = l2b_items
    sf = SelectFiles(str(root))
    table = sf.search_stac(tile='31TGJ', cc_max=30)
    assert list(table.columns) == ['satellite', 'level', 'tile', 'cloud_coverage', 'version', 'abspath', 'item']
    assert list(table.cloud_coverage) == [5., 30.] and list(table.item) == items[:2]
    assert table.index.is_monotonic_increasing and table.index[0] == pd.Timestamp('2025-01-03 10:33:40')
    # then selected by date as the file name listings
    files = sf.list_file_path(('2025-01-05', '2025-01-31'))
    assert list(files) == [items[1][:-5] + '.nc']
    assert len(SelectFiles(str(root)).search_stac(version='V3.1.0')) == 3
    assert SelectFiles(str(root)).search_stac(version='3.0.0').empty
    assert len(SelectFiles(str(root)).search_stac(date_startend=('2025-01-08', '2025-01-08'))) == 1


def test_selectfiles_search_stac_api(l2b_items):
    from grstbx import SelectFiles

    root, items = l2b_items
    features = []
    for path in items:
        with open(path) as f:
            item = json.load(f)
        item['assets']['data']['href'] = f'https://example.org/{item["id"]}.zarr'
        features.append(item)
    server, url = _serve_items(features)
    try:
        table = SelectFiles().search_stac(url, collections='grs-l2b', cc_max=50)
    finally:
        server.shutdown()
    assert len(table) == 2 and table.abspath.str.startswith('https://example.org/').all()


def test_dashboard_panel_inline(tmp_path):
    import panel as pn

    from grstbx.app import create_app, panel

    layout = panel(root=str(tmp_path), sidebar_width=300)
    assert isinstance(layout, pn.Row)
    sidebar, *main = layout
    assert sidebar.width == 300
    assert len(sidebar) == len(create_app(root=str(tmp_path)).sidebar)
    assert len(main) == 1
    layout.get_root()  # renders as in a notebook cell


def test_select_products_keeps_the_levels_apart():
    """The L2A and L2B products of an image are not two processings of the same product."""
    stamp = 'S2A_{}_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0.zarr'
    products = pd.concat([pd.DataFrame([product_info(f'/data/{stamp.format(level)}')])
                          for level in ('MSIL2AGRS', 'MSIL2B')], ignore_index=True)
    assert sorted(select_products(products).level) == ['L2A', 'L2B']


def test_open_products_refuses_mixed_levels(tmp_path, l2b_files):
    from conftest import l2b_product

    l2a = tmp_path / 'S2A_MSIL2AGRS_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0.zarr'
    ds = l2b_product('2025-01-03T10:33:39')[['mask']]
    ds['Rrs'] = ds['mask'].astype(float).expand_dims(wl=[443, 490])
    ds.to_zarr(l2a, consolidated=False)
    with pytest.raises(ValueError, match='cannot be displayed together'):
        open_products([str(l2a), l2b_files[0]])


def dashboard(**kwargs):
    from grstbx.app import create_app

    app = create_app(**kwargs)
    widgets = [w for part in app.sidebar for w in part.select(pn.widgets.Widget)]
    return app, lambda name: next(w for w in widgets if w.name == name)


def test_dashboard_failed_opening_keeps_the_viewer(l2b_files, monkeypatch):
    errors = []

    class Notifications:
        def error(self, message, **kwargs):
            errors.append(message)

    monkeypatch.setattr(type(pn.state), 'notifications', property(lambda self: Notifications()))
    app, widget = dashboard()
    widget('Products').value = l2b_files[0]
    widget('Open').clicks += 1
    opened = app.main[0].objects
    assert len(opened) == 3     # viewer, exports, search map

    widget('Products').value = '/no/such/product.zarr'
    widget('Open').clicks += 1
    assert errors and app.main[0].objects == opened


def test_dashboard_chooses_the_level_of_the_stac_results(l2b_files, monkeypatch):
    found = pd.DataFrame([{**product_info(path), 'level': level, 'item': path}
                          for path, level in zip(l2b_files, ['L2A', 'L2B', 'L2B'])])
    monkeypatch.setattr('grstbx.app.search_items', lambda *args, **kwargs: found)
    app, widget = dashboard(stac='/catalog/catalog.json')
    level = widget('Level to open')
    widget('Search').clicks += 1
    assert level.visible and level.options == ['L2A', 'L2B']
    widget('Open the results as a time series').clicks += 1     # by default, the last level
    assert widget('Products').value.splitlines() == l2b_files[1:]
    level.value = 'L2A'
    widget('Open the results as a time series').clicks += 1
    assert widget('Products').value.splitlines() == l2b_files[:1]

    monkeypatch.setattr('grstbx.app.search_items', lambda *args, **kwargs: found[found.level == 'L2B'])
    widget('Search').clicks += 1
    assert not level.visible
    widget('Open the results as a time series').clicks += 1
    assert widget('Products').value.splitlines() == l2b_files[1:]


def _mercator(*lonlat):
    import pyproj

    to_map = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    xs, ys = zip(*(to_map.transform(lon, lat) for lon, lat in lonlat))
    return list(xs), list(ys)


def test_search_map_area():
    from shapely.geometry import Point

    from grstbx.app import SearchMap

    search_map = SearchMap()
    assert search_map.geometry() is None
    xs, ys = _mercator((5.1, 43.5))
    search_map.pin_stream.event(data={'x': xs, 'y': ys})
    square = _mercator((4, 43), (4.5, 43), (4.5, 43.5), (4, 43.5))
    search_map.polygon_stream.event(data={'xs': [square[0], [0, 1]], 'ys': [square[1], [0, 1]]})  # 2nd unfinished
    bow_tie = _mercator((6, 44), (7, 45), (7, 44), (6, 45))      # lasso crossing itself
    search_map.lasso_stream.event(data={'xs': [bow_tie[0]], 'ys': [bow_tie[1]]})
    area = search_map.geometry()
    assert area.is_valid
    for lon, lat in [(5.1, 43.5), (4.2, 43.2), (6.2, 44.5), (6.8, 44.5)]:   # both lobes of the bow tie
        assert area.buffer(1e-9).contains(Point(lon, lat))
    assert not area.intersects(Point(5.5, 43.2)) and not area.intersects(Point(6.5, 44.2))
    search_map.clear()
    assert search_map.geometry() is None


def test_dashboard_searches_the_area_of_the_search_map(l2b_files, monkeypatch):
    from grstbx import app as app_module

    maps, calls = [], []

    class SearchMap(app_module.SearchMap):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            maps.append(self)

    monkeypatch.setattr(app_module, 'SearchMap', SearchMap)
    found = pd.DataFrame([{**product_info(path), 'item': path} for path in l2b_files])
    monkeypatch.setattr(app_module, 'search_items', lambda *args, **kwargs: calls.append(kwargs) or found)
    app, widget = dashboard(stac='/catalog/catalog.json')
    area = next(w for part in app.sidebar for w in part.select(pn.widgets.RadioButtonGroup)
                if 'Search map' in w.options)
    assert [getattr(o, 'title', None) for o in app.main[0].objects][1:] == ['Area of the STAC search',
                                                                       'How to use the viewer']

    area.value = 'Search map'
    widget('Search').clicks += 1        # nothing drawn: no search
    assert not calls
    area.value = 'Everywhere'
    xs, ys = _mercator((5.1, 43.5))
    maps[0].pin_stream.event(data={'x': xs, 'y': ys})
    assert area.value == 'Search map'   # selected by the drawing
    widget('Search').clicks += 1
    assert calls[-1]['intersects'].equals(maps[0].geometry())
    assert not widget('Open the results as a time series').disabled


def _tile_product(folder, tile, stamp, chla, crs='EPSG:32631', x0=700010, y0=4899990):
    """L2B NetCDF product of a tile: 40 x 40 pixels of 20 m from (x0, y0), the left quarter without data."""
    from conftest import l2b_product

    ds = l2b_product(pd.Timestamp(stamp), chla, size=40)
    ds = ds.assign_coords(x=x0 + 20 * np.arange(40), y=y0 - 20 * np.arange(40)).rio.write_crs(crs)
    path = folder / f'S2A_MSIL2B_{stamp}_N0511_R108_T{tile}_{stamp[:8]}T150000_V3.1.0.nc'
    ds.to_netcdf(path)
    return str(path)


def test_open_area_combines_the_tiles_of_an_acquisition(tmp_path):
    import pyproj
    from shapely.geometry import box

    # 31TGJ and 31TGK (same UTM zone, the second 400 m to the east), and 32TLP (zone 32) further east, all
    # of the same acquisition; 31TGJ alone the next day
    to32 = pyproj.Transformer.from_crs(32631, 32632, always_xy=True)
    x32, y32 = (round(v / 20) * 20 + 10 for v in to32.transform(701610, 4899990))
    paths = [_tile_product(tmp_path, '31TGJ', '20250103T103339', 1.),
             _tile_product(tmp_path, '31TGK', '20250103T103339', 2., x0=700410),
             _tile_product(tmp_path, '32TLP', '20250103T103339', 3., crs='EPSG:32632', x0=x32, y0=y32),
             _tile_product(tmp_path, '31TGJ', '20250108T103421', 4.)]
    with pytest.raises(SeveralZones):
        open_products(paths)
    to_lonlat = pyproj.Transformer.from_crs(32631, 4326, always_xy=True)
    area = box(*to_lonlat.transform(700000, 4899200), *to_lonlat.transform(702400, 4900000))
    data, kind, params, title = open_products(paths, area=area)
    assert kind == 'L2B' and len(data.images) == 2 and '2 L2B images of the area' in title
    assert data.crs == 'EPSG:32631' and [len(image.paths) for image in data.images] == [3, 1]

    image = data.images[0][0]['Chla_OC2nasa']
    assert image.x[1] - image.x[0] == 20 and float(image.x[0]) % 20 == 10     # on the grid of the zone
    at = lambda x, y: float(image.sel(x=x, y=y, method='nearest'))            # noqa: E731
    y = 4899990 - 20 * 20
    assert at(700010 + 20 * 15, y) == 1          # 31TGJ
    assert at(700410 + 20 * 39, y) == 2          # 31TGK only
    assert at(701610 + 20 * 30, y) == 3          # 32TLP only, reprojected
    assert np.isnan(at(700010 + 20 * 2, y))      # no data in 31TGJ, outside 31TGK
    assert np.nanmax(data.images[1][0]['Chla_OC2nasa'].values) == 4
    # a pin opens the area around it
    pin = box(*to_lonlat.transform(700500, 4899500), *to_lonlat.transform(700500, 4899500))
    data, *_ = open_products(paths[:2], area=pin)
    grid = data.images[0].grid
    assert abs(len(grid['x']) * 20 - 5000) <= 40 and abs(len(grid['y']) * 20 - 5000) <= 40


def test_open_area_refuses_large_areas(l2b_files):
    with pytest.raises(ValueError, match='too large'):
        open_products(l2b_files, area=(0., 40., 5., 45.))


def test_dashboard_opens_the_area_of_the_search(l2b_files, monkeypatch):
    from grstbx import app as app_module

    found = pd.DataFrame([{**product_info(path), 'item': path} for path in l2b_files])
    opened = []
    monkeypatch.setattr(app_module, 'search_items', lambda *args, **kwargs: found)
    monkeypatch.setattr(app_module, 'open_products', lambda *args, **kwargs: opened.append(kwargs) or 1 / 0)
    app, widget = dashboard(stac='/catalog/catalog.json')
    area = next(w for part in app.sidebar for w in part.select(pn.widgets.RadioButtonGroup)
                if 'Bounding box' in w.options)
    clip = widget('Open the area only (tiles clipped and combined)')
    widget('Search').clicks += 1                     # everywhere: nothing to clip
    assert not clip.visible
    widget('Open the results as a time series').clicks += 1
    assert opened[-1]['area'] is None

    area.value = 'Bounding box'
    widget('Bounding box: lon_min, lat_min, lon_max, lat_max (deg)').value = '5.0, 43.4, 5.2, 43.5'
    widget('Search').clicks += 1
    assert clip.visible and clip.value
    widget('Open the results as a time series').clicks += 1
    assert opened[-1]['area'] == [5.0, 43.4, 5.2, 43.5]
    clip.value = False
    widget('Open the results as a time series').clicks += 1
    assert opened[-1]['area'] is None


def test_search_map_starts_on_the_area():
    import pyproj

    from grstbx.app import SEARCH_AREA, SearchMap

    to_lonlat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
    search_map = SearchMap(width=800, height=400)
    (x0, x1), (y0, y1) = search_map.extent()
    assert abs((x1 - x0) / (y1 - y0) - 2) < 1e-9                   # square tiles
    lon0, lat0 = to_lonlat.transform(x0, y0)
    lon1, lat1 = to_lonlat.transform(x1, y1)
    lon_min, lat_min, lon_max, lat_max = SEARCH_AREA
    assert lon0 <= lon_min and lon1 >= lon_max and lat0 <= lat_min + 1e-9 and lat1 >= lat_max - 1e-9
    assert lon1 - lon0 < 2                                          # around the Etang de Berre
    (x0, x1), _ = SearchMap(width=800, height=400, area=None).extent()
    assert x1 - x0 == 2 * SearchMap.WORLD


def _pyramid_product(folder, tile, stamp, chla, crs='EPSG:32631', x0=700010, y0=4899990, size=64, nlevels=3,
                     grs_layout=False):
    """
    L2B Zarr pyramid of a tile (levels '0', '1', ... of 20, 40, ... m), the left quarter without data;
    ``grs_layout``: ``multiscales`` attribute as written by GRS (list, ``group`` and ``scale`` of each level)
    instead of GRSl2bgen.
    """
    import zarr
    from conftest import l2b_product

    ds = l2b_product(pd.Timestamp(stamp), chla, size=size).drop_vars(['time', 'forecast_period'])
    ds = ds.assign_coords(x=x0 + 20. * np.arange(size), y=y0 - 20. * np.arange(size)).rio.write_crs(crs)
    path = folder / f'S2A_MSIL2B_{stamp}_N0511_R108_T{tile}_{stamp[:8]}T150000_V3.1.0.zarr'
    level = ds
    for index in range(nlevels):
        level.to_zarr(path, group=str(index), consolidated=False)
        level = level.coarsen(x=2, y=2).mean().rio.write_crs(crs)
    layout = [{'asset': '0'}] + [{'asset': str(i), 'derived_from': str(i - 1), 'transform': {'scale': [2.0, 2.0]}}
                                 for i in range(1, nlevels)]
    multiscales = {'layout': layout}
    if grs_layout:
        multiscales = [{'layout': [{'group': str(i), 'scale': 2 ** i} for i in range(nlevels)]}]
    zarr.open_group(path, mode='a').attrs.update(
        {'multiscales': multiscales, 'acquisition_date': pd.Timestamp(stamp).isoformat()})
    return str(path)


@pytest.mark.parametrize('grs_layout', [False, True])
def test_open_pyramid_levels(tmp_path, grs_layout):
    """Levels and resolutions of the pyramids of GRS (list) and GRSl2bgen (dict) ``multiscales``."""
    path = _pyramid_product(tmp_path, '31TGJ', '20250103T103339', 1., grs_layout=grs_layout)
    data, kind, params, title = open_products(path)
    image = data.images[0]
    assert len(image) == 3 and [image.resolution(level) for level in range(3)] == [None, 40., 80.]


def test_open_area_from_the_pyramids(tmp_path, monkeypatch):
    import pyproj
    from shapely.geometry import box

    from grstbx import app as app_module

    to32 = pyproj.Transformer.from_crs(32631, 32632, always_xy=True)
    x32, y32 = (round(v / 20) * 20 + 10 for v in to32.transform(701290, 4899990))
    paths = [_pyramid_product(tmp_path, '31TGJ', '20250103T103339', 1.),
             _pyramid_product(tmp_path, '32TLP', '20250103T103339', 3., crs='EPSG:32632', x0=x32, y0=y32)]
    to_lonlat = pyproj.Transformer.from_crs(32631, 4326, always_xy=True)
    area = box(*to_lonlat.transform(700000, 4898700), *to_lonlat.transform(702500, 4900000))
    # beyond the limit of the products without pyramid: the pyramids are used
    monkeypatch.setattr(app_module, 'AREA_MAX_PIXELS', 50)
    data, kind, params, title = open_products(paths, area=area)
    image = data.images[0]
    assert len(image) == 3 and [data.resolution(0, level) for level in range(3)] == [20., 40., 80.]
    assert data.nlevels(0) == 3
    for level in range(3):
        layer = image[level]['Chla_OC2nasa']
        res = 20 * 2 ** level
        assert float(layer.x[1] - layer.x[0]) == res and layer.chunks is not None     # lazy
        at = lambda x: float(layer.sel(x=x, y=4899990 - 20 * 30, method='nearest'))   # noqa: E731
        assert at(700010 + 20 * 40) == 1                 # 31TGJ
        assert at(701290 + 20 * 50) == 3                 # 32TLP only, reprojected block by block
        assert np.isnan(at(700010 + 20 * 4))             # no data in 31TGJ

    with pytest.raises(ValueError, match='without pyramid'):
        open_products([_tile_product(tmp_path, '31TGJ', '20250108T103421', 4.)], area=area)


def test_open_area_coarser_levels_and_incomplete_products(tmp_path, monkeypatch):
    """Levels computed beyond the pyramids of a large area; the products without coordinates are left out."""
    import pyproj
    import zarr
    from shapely.geometry import box

    from grstbx import app as app_module

    good = _pyramid_product(tmp_path, '31TGJ', '20250103T103339', 1.)
    broken = _pyramid_product(tmp_path, '31TFJ', '20250103T103339', 2., x0=700010 - 64 * 20, grs_layout=True)
    for level in range(3):      # incomplete GRS store: coordinates never written
        group = zarr.open_group(broken, path=str(level), mode='a')
        group['x'][:] = np.nan
    to_lonlat = pyproj.Transformer.from_crs(32631, 4326, always_xy=True)
    area = box(*to_lonlat.transform(690000, 4880000), *to_lonlat.transform(720000, 4910000))
    monkeypatch.setattr(app_module, 'AREA_MAX_PIXELS', 50)
    data, kind, params, title = open_products([broken, good], area=area, min_size=64)
    image = data.images[0]
    # 1500 pixels of 20 m, coarsest stored level of 80 m (375 pixels): 3 levels computed down to 64 pixels
    assert len(image) == 3 and data.nlevels(0) == 6
    assert [data.resolution(0, level) for level in range(6)] == [20., 40., 80., 160., 320., 640.]
    layer = data.layer(0, 'Chla_OC2nasa', 5)
    assert layer.sizes['x'] <= 64 and np.nanmax(layer.values) == 1      # only the complete product
    da, level = data.window(0, 'Chla_OC2nasa', data.bounds(0), 30000 / 50)
    assert level == 4

    with pytest.raises(ValueError, match='no product of the area has coordinates'):
        open_products([broken], area=area)


def test_warp_reads_only_the_blocks_displayed(tmp_path, monkeypatch):
    import pyproj
    import xarray as xr

    from grstbx import app as app_module

    to32 = pyproj.Transformer.from_crs(32631, 32632, always_xy=True)
    x32, y32 = (round(v / 20) * 20 + 10 for v in to32.transform(700010, 4899990))
    path = _pyramid_product(tmp_path, '32TLP', '20250103T103339', 3., crs='EPSG:32632', x0=x32, y0=y32, size=128,
                            nlevels=1)
    grid = app_module._area_grid((700000, 4897400, 702560, 4900000), pyproj.CRS(32631), 20.)
    layer = app_module._LazyLevels(path, 'L2B', ['Chla_OC2nasa'])[0]['Chla_OC2nasa']
    reads = []
    original = xr.DataArray.compute
    monkeypatch.setattr(xr.DataArray, 'compute', lambda self, **kw: reads.append(self.shape) or original(self, **kw))
    warped = app_module._warp(layer, grid, block=32)
    assert warped.shape == (len(grid['y']), len(grid['x'])) and not reads
    block = warped.isel(x=slice(32, 64), y=slice(32, 64)).values     # one block of the grid
    assert len(reads) == 1 and max(reads[0]) < 50 and np.nanmax(block) == 3
