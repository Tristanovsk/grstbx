import numpy as np
import pytest

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


def _flag_product(path, nodata_fraction=0.0, seed=0):
    """Small L2A Zarr product (Rrs, mask, flags with flag_names), UTM 31N grid, 4 chunks per side."""
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(seed)
    n = 40
    flags = rng.integers(0, 16, size=(n, n)).astype('uint32')      # bits 0-3 raised at random
    flags[:int(n * nodata_fraction)] |= 1                           # bit 0: nodata, on the first rows
    ds = xr.Dataset({'Rrs': (('wl', 'y', 'x'), rng.random((2, n, n))),
                     'mask': (('y', 'x'), (flags & 1).astype('uint8')),
                     'flags': (('y', 'x'), flags, {'flag_names': 'nodata cloud water None shadow'})},
                    coords={'wl': [490, 560], 'y': 4800000. - 20 * np.arange(n), 'x': 600000. + 20 * np.arange(n)})
    ds = ds.rio.write_crs(32631)
    ds.chunk({'y': 10, 'x': 10}).to_zarr(path, consolidated=True)
    return path


def _reference_stats(flags):
    import numpy as np

    names = flags.attrs['flag_names'].split(' ')
    values = np.asarray(flags)
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    values = values.astype(np.int64).ravel()
    return {f'flag_{name}': ((values >> bit) & 1).mean() for bit, name in enumerate(names) if name not in ('None', '')}


def test_flag_stats_by_chunk(tmp_path):
    import numpy as np
    import xarray as xr

    from grstbx.driver import L2grs, open_zarr_image

    flags = open_zarr_image(str(_flag_product(tmp_path / 'p.zarr')))['flags']
    assert flags.chunks is not None and len(flags.chunks[0]) == 4
    expected = _reference_stats(flags)
    for raster in (flags, flags.load()):                    # lazy (chunk by chunk) and in memory
        stats = L2grs.get_flag_stats(raster.expand_dims(time=[np.datetime64('2025-08-01')]))
        assert set(stats.data_vars) == set(expected) and 'flag_None' not in stats
        for name, value in expected.items():
            assert float(stats[name].values[0]) == pytest.approx(value)
    # float bitmask with NaN (e.g. after a reprojection): the NaN pixels are not counted
    floating = flags.astype('float64').where(flags.x < flags.x[30]).chunk({'y': 10, 'x': 10})
    floating.attrs = flags.attrs
    stats = L2grs.get_flag_stats(floating.expand_dims(time=[np.datetime64('2025-08-01')]))
    for name, value in _reference_stats(floating).items():
        assert float(stats[name].values[0]) == pytest.approx(value)
    # several dates in one array: one value per date
    two = xr.concat([flags, flags], dim='time').assign_coords(time=np.array(['2025-08-01', '2025-08-02'], 'M8[ns]'))
    assert L2grs.get_flag_stats(two)['flag_cloud'].values.tolist() == [pytest.approx(expected['flag_cloud'])] * 2


def test_l2a_datacube_subset_loaded_once(tmp_path):
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    from grstbx.driver import L2grs

    files = [str(_flag_product(tmp_path / f'S2A_MSIL2AGRS_2025080{k}T103000_{k}.zarr', nodata_fraction=nd, seed=k))
             for k, nd in [(1, 0.0), (2, 0.9), (3, 0.1)]]
    aoi = gpd.GeoDataFrame(geometry=[box(600100, 4799300, 600500, 4799900)], crs=32631).to_crs(4326)
    loaded = L2grs(files)
    loaded.get_l2a_datacube(subset=aoi, nodata_thresh=0.5)
    lazy = L2grs(files)
    lazy.get_l2a_datacube(subset=aoi, nodata_thresh=0.5, load=False)
    cube = loaded.datacube
    assert cube.sizes['time'] == 2                           # the image with 90 % nodata dropped
    assert cube.Rrs.chunks is None and lazy.datacube.Rrs.chunks is not None
    assert cube.Rrs.dtype == np.float32 and 0 < cube.sizes['x'] < 40
    assert cube.equals(lazy.datacube.load())
    assert 'flag_cloud' in cube and cube.flag_nodata.dims == ('time',)
