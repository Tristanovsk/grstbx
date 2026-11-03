"""Static STAC catalogue of products, read from disk and from a local S3 service (rclone serve s3)."""
import contextlib
import json
import os
import shutil
import socket
import subprocess
import time

import pandas as pd
import pytest
import xarray as xr

from conftest import l2b_product
from grstbx import batch, stac
from grstbx.app import _static_items, open_products, search_items
from grstbx.datalake import SelectFiles

BUCKET = 's3://bucket'
ATTRS = {'constellation': 'Sentinel-2', 'instrument': 'MSI', 'tile': '31TGJ', 'version': '3.1.0',
         'DATATAKE_1_SPACECRAFT_NAME': 'Sentinel-2A', 'CLOUD_COVERAGE_ASSESSMENT': '12.5',
         'acquisition_date': '2025-01-03T10:33:39.024Z', 'processing_time': '2026-10-16T22:40:03.609653'}


def l2a_store(root, stamp, cloud='12.5', complete=True):
    """L2A Zarr store (pyramid of two levels) in the tree <tile>/YYYY/MM/DD/<product>/<product>.zarr."""
    name = f'S2A_MSIL2AGRS_{stamp}_N0511_R108_T31TGJ_{stamp[:8]}T150000_V3.1.0'
    time_ = pd.Timestamp(stamp)
    folder = root / '31TGJ' / f'{time_:%Y}' / f'{time_:%m}' / f'{time_:%d}' / name
    folder.mkdir(parents=True)
    (folder / 'log_file.log').write_text('log')
    ds = l2b_product(time_)[['mask']]
    ds['Rrs'] = ds['mask'].astype(float).expand_dims(wl=[443, 490])
    store = folder / f'{name}.zarr'
    attrs = {**ATTRS, 'acquisition_date': f'{time_.isoformat()}Z', 'CLOUD_COVERAGE_ASSESSMENT': cloud}
    xr.Dataset(attrs=attrs).to_zarr(store, mode='w', consolidated=False)
    for group in ['0', '1']:
        ds.to_zarr(store, group=group, mode='a', consolidated=False)
    if not complete:
        os.remove(store / '0' / 'mask' / 'zarr.json')
    return store


def l2b_folder(root, l2a_store_path, stamp):
    """L2B product with its Item, as written by GRSl2bgen --product_dir --stac."""
    name = os.path.basename(l2a_store_path)[:-len('.zarr')].replace('_MSIL2AGRS_', '_MSIL2B_')
    time_ = pd.Timestamp(stamp)
    folder = root / '31TGJ' / f'{time_:%Y}' / f'{time_:%m}' / f'{time_:%d}' / name
    folder.mkdir(parents=True)
    l2b_product(time_).to_zarr(folder / f'{name}.zarr', mode='w', consolidated=False)
    (folder / f'{name}.log').write_text('log')
    item = {'type': 'Feature', 'stac_version': '1.0.0', 'id': name, 'bbox': [4.2, 43.2, 5.6, 44.2],
            'geometry': None, 'links': [{'rel': 'derived_from', 'href': '/data/L2A/local.zarr'}],
            'properties': {'datetime': f'{time_.isoformat()}Z', 'processing:level': 'L2B', 'eo:cloud_cover': 5.0,
                           'grid:code': 'MGRS-31TGJ', 'processing:software': {'GRS': '3.1.0'}},
            'assets': {'data': {'href': f'./{name}.zarr', 'roles': ['data']},
                       'log': {'href': f'./{name}.log', 'roles': ['metadata']}}}
    (folder / f'{name}.json').write_text(json.dumps(item))


@pytest.fixture
def products(tmp_path):
    """Folders of a complete and an incomplete L2A, an L2B with Item, and a loose L2B NetCDF file."""
    l2a, l2b = tmp_path / 'L2A', tmp_path / 'L2B'
    store = l2a_store(l2a, '20250103T103339')
    l2a_store(l2a, '20250108T103421', complete=False)
    l2b_folder(l2b, store, '20250103T103339')
    folder = l2b / '31TGJ' / '2025' / '01' / '13'
    folder.mkdir(parents=True)
    name = 'S2A_MSIL2B_20250113T103309_N0511_R108_T31TGJ_20250113T150000_V3.1.0'
    l2b_product('2025-01-13T10:33:09').assign_attrs(
        {**ATTRS, 'acquisition_date': '2025-01-13T10:33:09.024Z'}).to_netcdf(folder / f'{name}.nc')
    return {l2a: f'{BUCKET}/L2A', l2b: f'{BUCKET}/L2B'}


@pytest.fixture
def catalog(tmp_path, products):
    out = tmp_path / 'bucket' / 'stac'
    counts, skipped = stac.build_catalog({str(k): v for k, v in products.items()}, str(out))
    return out, counts, skipped


def test_catalog_content(catalog):
    out, counts, skipped = catalog
    assert counts == {'grs-l2a': 1, 'grs-l2b': 2}
    assert len(skipped) == 1 and '20250108T103421' in skipped[0]

    root = json.loads((out / 'catalog.json').read_text())
    assert [link['href'] for link in root['links'] if link['rel'] == 'child'] == \
        ['./grs-l2a/collection.json', './grs-l2b/collection.json']
    collection = json.loads((out / 'grs-l2a' / 'collection.json').read_text())
    assert collection['extent']['temporal']['interval'][0][0].startswith('2025-01-03')
    assert collection['summaries']['grid:code'] == ['MGRS-31TGJ']


def test_l2a_item_from_attributes(catalog):
    out = catalog[0]
    name = 'S2A_MSIL2AGRS_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0'
    item = json.loads((out / 'grs-l2a' / name / f'{name}.json').read_text())
    props = item['properties']
    assert item['collection'] == 'grs-l2a'
    assert props['processing:level'] == 'L2A' and props['processing:software'] == {'GRS': '3.1.0'}
    assert props['eo:cloud_cover'] == 12.5 and props['grid:code'] == 'MGRS-31TGJ'
    assert props['proj:code'] == 'EPSG:32631' and props['proj:shape'] == [20, 20]
    assert props['platform'] == 'sentinel-2a'
    assert item['assets']['data']['href'] == f"{BUCKET}/L2A/31TGJ/2025/01/03/{name}/{name}.zarr"
    assert item['assets']['log']['href'].endswith(f'{name}/log_file.log')
    ring = item['geometry']['coordinates'][0]
    assert ring[0] == ring[-1] and len(ring) == 5


def test_l2b_item_rewritten_and_linked_to_l2a(catalog):
    out = catalog[0]
    name = 'S2A_MSIL2B_20250103T103339_N0511_R108_T31TGJ_20250103T150000_V3.1.0'
    item = json.loads((out / 'grs-l2b' / name / f'{name}.json').read_text())
    assert item['assets']['data']['href'].startswith(f'{BUCKET}/L2B/31TGJ/')
    link = next(link for link in item['links'] if link['rel'] == 'derived_from')
    assert (out / 'grs-l2b' / name / link['href']).resolve().exists()      # no more the local L2A path


def test_loose_netcdf_gets_an_item(catalog):
    items = json.loads((catalog[0] / 'grs-l2b' / 'items.json').read_text())['features']
    item = next(item for item in items if '20250113' in item['id'])
    assert item['assets']['data']['type'] == 'application/x-netcdf'
    assert item['assets']['data']['file:size'] > 0
    assert not any(link['rel'] == 'derived_from' for link in item['links'])     # no L2A in the catalogue


def test_item_links_resolve_in_the_catalogue(catalog):
    """Relative links of the Items, in their own file and in items.json."""
    out = catalog[0]
    for collection in ('grs-l2a', 'grs-l2b'):
        for item in json.loads((out / collection / 'items.json').read_text())['features']:
            for link in item['links']:
                assert (out / collection / item['id'] / link['href']).resolve().exists()


def test_search_static_catalogue_on_disk(catalog):
    source = str(catalog[0] / 'catalog.json')
    assert len(search_items(source)) == 3
    assert len(search_items(source, collections='grs-l2b')) == 2
    assert len(search_items(source, collections=['grs-l2a'], max_cloud_cover=10)) == 0
    found = search_items(source, start='2025-01-10')
    assert list(found.level) == ['L2B']
    # every Item has its location, even when read from items.json
    assert all(os.path.isfile(item) for item in search_items(source).item)


def test_static_catalogue_without_items_file(catalog):
    """A static catalogue of another origin: Items read one by one from the `item` links."""
    out = catalog[0]
    os.remove(out / 'grs-l2b' / 'items.json')
    collection_file = out / 'grs-l2b' / 'collection.json'
    collection = json.loads(collection_file.read_text())
    collection['links'] = [link for link in collection['links'] if link['rel'] != 'items']
    collection_file.write_text(json.dumps(collection))
    assert len(_static_items(str(out / 'catalog.json'), 'grs-l2b')) == 2
    assert len(_static_items(str(collection_file))) == 2


def test_collection_file_as_source(catalog):
    assert len(search_items(str(catalog[0] / 'grs-l2b' / 'collection.json'))) == 2


@contextlib.contextmanager
def serve_s3(root, tmp_path, monkeypatch, read_only=True):
    """Folder served as buckets by `rclone serve s3` on a free port, credentials and endpoint in a profile."""
    rclone = shutil.which('rclone')
    if rclone is None:
        pytest.skip('rclone is not installed')
    s3fs = pytest.importorskip('s3fs')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    (tmp_path / 'credentials').write_text('[test]\naws_access_key_id = testkey\naws_secret_access_key = testsecret\n')
    (tmp_path / 'config').write_text(f'[profile test]\nregion = us-east-1\nendpoint_url = http://127.0.0.1:{port}\n')
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(tmp_path / 'credentials'))
    monkeypatch.setenv('AWS_CONFIG_FILE', str(tmp_path / 'config'))
    monkeypatch.setenv('AWS_PROFILE', 'test')
    s3fs.S3FileSystem.clear_instance_cache()
    process = subprocess.Popen([rclone, 'serve', 's3', str(root), '--addr', f'127.0.0.1:{port}', '--auth-key',
                                'testkey,testsecret', '--etag-hash', ''] + (['--read-only'] if read_only else []),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                break
        time.sleep(0.05)
    try:
        yield
    finally:
        process.terminate()
        process.wait()
        s3fs.S3FileSystem.clear_instance_cache()


@pytest.fixture
def s3_server(tmp_path, products, catalog, monkeypatch):
    """Bucket of the products and of their catalogue, read-only."""
    bucket = tmp_path / 'bucket'
    for folder in products:
        shutil.copytree(folder, bucket / folder.name)
    with serve_s3(tmp_path, tmp_path, monkeypatch):
        yield


def test_search_and_open_from_bucket(s3_server):
    source = f'{BUCKET}/stac/catalog.json'
    found = search_items(source, collections='grs-l2a')
    assert len(found) == 1 and found.path[0].endswith('.zarr') and found.item[0].startswith('s3://')
    data, kind, *_ = open_products(list(found.path))
    assert kind == 'L2A'
    assert data is not None


def test_item_url_in_bucket(s3_server):
    from grstbx.app import catalogue

    found = search_items(f'{BUCKET}/stac/catalog.json', collections='grs-l2a')
    assert catalogue(list(found.item)).path[0] == found.path[0]


def test_netcdf_product_from_bucket(s3_server):
    found = search_items(f'{BUCKET}/stac/catalog.json', collections='grs-l2b', start='2025-01-10')
    data, kind, params, _ = open_products(list(found.path))
    assert kind == 'L2B' and 'Chla_OC2nasa' in params


def test_select_files_from_bucket(s3_server):
    listing = SelectFiles('.').search_stac(f'{BUCKET}/stac/catalog.json', collections='grs-l2b', cc_max=10)
    assert list(listing.level) == ['L2B'] and listing.abspath.iloc[0].startswith('s3://')


def test_wrong_credentials_are_rejected(s3_server):
    import s3fs

    fs = s3fs.S3FileSystem(key='testkey', secret='wrong', skip_instance_cache=True)
    with pytest.raises((PermissionError, OSError)):
        fs.ls('bucket')


def test_batch_and_catalogue_in_bucket(tmp_path, monkeypatch):
    """L2A products in a bucket -> L2B products written to the bucket by a batch, catalogue built from their Items."""
    local = tmp_path / 'local'
    store = l2a_store(local, '20250103T103339')
    assert batch.write_l2a_item(str(store)) == str(store)[:-len('.zarr')] + '.json'
    root = tmp_path / 'root'
    shutil.copytree(local, root / 'bucket' / 'L2A')

    def run_grsl2bgen(l2a, l2b, l2a_href=None, **kwargs):
        """GRSl2bgen stand-in: L2B product of the local copy of the L2A product, with its Item."""
        assert os.path.isdir(l2a) and l2a_href.startswith(f'{BUCKET}/L2A/')
        name = os.path.basename(l2b)[:-len('.zarr')]
        out = tmp_path / 'l2b'
        l2b_folder(out, os.path.basename(l2a) + '.zarr', '20250103T103339')
        shutil.copytree(next(out.rglob(name)), os.path.dirname(l2b), dirs_exist_ok=True)

    monkeypatch.setattr(batch, 'run_grsl2bgen', run_grsl2bgen)
    with serve_s3(root, tmp_path, monkeypatch, read_only=False):
        products = batch.list_l2a_products(f'{BUCKET}/L2A')
        assert products['complete'].all() and products['path'].iloc[0].startswith(BUCKET)
        run = batch.Batch(l2b_dir=f'{BUCKET}/L2B', l2b_ext='.zarr', work_dir=str(tmp_path / 'work'))
        plan = run.plan_l2a(products)
        assert run.run(plan, verbose=False)['status'].tolist() == ['done']
        assert run.plan_l2a(products)['l2b_done'].all() and batch.read_profiles(plan['profile'])['status'].tolist() == ['reused', 'done']

        counts, skipped = stac.build_catalog([f'{BUCKET}/L2A', f'{BUCKET}/L2B'], f'{BUCKET}/stac')
        assert counts == {'grs-l2a': 1, 'grs-l2b': 1} and skipped == []
        found = search_items(f'{BUCKET}/stac/catalog.json')
        assert sorted(found.level) == ['L2A', 'L2B'] and all(p.startswith(f'{BUCKET}/L') for p in found.path)
        l2b_item = json.loads(batch._read_text(f'{BUCKET}/stac/grs-l2b/items.json'))['features'][0]
        assert any(link['rel'] == 'derived_from' for link in l2b_item['links'])
        data, kind, *_ = open_products([p for p in found.path if 'L2B' in p])
        assert kind == 'L2B'
