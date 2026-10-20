import json
import os
import shutil

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
