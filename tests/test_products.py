import pandas as pd

from grstbx.app import find_products, product_info, select_products


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
