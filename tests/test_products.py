import os
import shutil

import pandas as pd

from grstbx.app import _product_file, catalogue, find_products, is_product, product_info, select_products


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
