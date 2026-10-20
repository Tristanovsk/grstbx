import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from grstbx.app import (_product_file, catalogue, find_products, is_product, is_stac_item, product_info,
                        select_products)


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
