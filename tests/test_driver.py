import numpy as np

import grstbx


def test_get_l2b_datacube(l2b_files):
    aoi = grstbx.aoi_from_box(*_centre(), width=200, height=200)
    dc = grstbx.L2grs(l2b_files)
    dc.get_l2b_datacube(subset=aoi, var='Chla_OC2nasa', variables=['mask'])
    cube = dc.datacube
    # the image without any valid pixel is skipped, the mixed float / timedelta forecast_period is not an issue
    assert cube.sizes['time'] == 2
    assert set(cube.data_vars) == {'Chla_OC2nasa', 'mask', 'valid_pix_prop'}
    assert cube.sizes['x'] < 20 and cube.sizes['y'] < 20
    np.testing.assert_allclose(cube.Chla_OC2nasa.median(['x', 'y']), [1.0, 2.0])


def test_get_l2b_datacube_all_variables(l2b_files):
    dc = grstbx.L2grs(l2b_files)
    dc.get_l2b_datacube()
    assert {'Chla_OC2nasa', 'TURB_dogliotti', 'flags', 'mask'} <= set(dc.datacube.data_vars)
    assert dc.datacube.valid_pix_prop.values.tolist() == [0.75, 0.75]


def _centre():
    from pyproj import Transformer

    return Transformer.from_crs(32631, 4326, always_xy=True).transform(700200, 4899800)
