from conftest import l2b_product
from grstbx import Masking


def test_masking():
    product = l2b_product('2025-01-03')
    masking = Masking(product)
    assert list(masking.print_info().index) == ['nodata', 'cloud', 'water']
    water = masking.get_mask(water=True, nodata=False)
    assert int(water.sum()) == int((product.mask == 0).sum()) == 300
