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


def test_to_float32_keeps_values_and_encoding(tmp_path):
    import numpy as np
    import xarray as xr

    from grstbx.driver import L2grs, open_zarr_image, to_float32

    values = np.arange(-300, 300, dtype='int16').reshape(1, 20, 30)
    ds = xr.Dataset({'Rrs': (('wl', 'y', 'x'), values * 2e-5), 'mask': (('y', 'x'), np.zeros((20, 30), 'uint8'))},
                    coords={'wl': [560], 'y': np.arange(20.), 'x': np.arange(30.)})
    encoding = {'Rrs': {'dtype': 'int16', 'scale_factor': 2e-5, '_FillValue': -32768}}
    ds.to_zarr(tmp_path / 'p.zarr', encoding=encoding, consolidated=True)
    ds.to_netcdf(tmp_path / 'p.nc', encoding=encoding)

    as64 = open_zarr_image(str(tmp_path / 'p.zarr'), float32=False)
    as32 = open_zarr_image(str(tmp_path / 'p.zarr'))
    assert as64.Rrs.dtype == np.float64 and as32.Rrs.dtype == np.float32 and as32['mask'].dtype == np.uint8
    assert as32.x.dtype == np.float64                                  # coordinates unchanged
    assert np.allclose(as32.Rrs.values, as64.Rrs.values, rtol=1e-6, atol=0)
    assert as32.Rrs.encoding['scale_factor'] == 2e-5                   # packing kept for the exports
    assert as32.Rrs.chunks is not None                                 # still lazy
    nc = to_float32(xr.open_dataset(tmp_path / 'p.nc', chunks={}))
    assert nc.Rrs.dtype == np.float32 and to_float32(nc.Rrs).dtype == np.float32 and to_float32(None) is None
    assert L2grs([str(tmp_path / 'p.zarr')]).load_l2b_image(str(tmp_path / 'p.zarr')).Rrs.dtype == np.float32
    assert L2grs([], float32=False).load_l2b_image(str(tmp_path / 'p.zarr')).Rrs.dtype == np.float64


def test_import_is_lazy_and_rio_works():
    import subprocess
    import sys

    code = ("import sys, numpy as np, xarray as xr, grstbx\n"
            "assert 'rioxarray' not in sys.modules and 'dask' not in sys.modules and 'rasterio' not in sys.modules\n"
            "da = xr.DataArray(np.zeros((2, 2)), dims=('y', 'x'), coords={'y': [1., 0.], 'x': [0., 1.]})\n"
            "assert str(da.rio.write_crs(32631).rio.crs) == 'EPSG:32631'\n"
            "assert type(da.to_dataset(name='a').rio).__name__ == 'RasterDataset'\n"
            "assert 'rioxarray' in sys.modules\n")
    result = subprocess.run([sys.executable, '-W', 'error::UserWarning', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
