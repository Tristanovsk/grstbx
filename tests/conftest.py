"""Synthetic GRS products used by the tests (no external data needed)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import rioxarray  # noqa: F401, activates the .rio accessor

FLAG_NAMES = ['nodata', 'cloud', 'water']


def l2b_product(time, chla=1.0, forecast_period=4.32, size=20):
    """L2B-like dataset on a 20 m UTM 31N grid, with flags, mask and two parameters."""
    x = 700010 + 20 * np.arange(size)
    y = 4900000 - 10 - 20 * np.arange(size)
    shape = (size, size)
    flags = np.full(shape, 4, dtype=np.uint32)             # water
    flags[:, : size // 4] = 1                               # nodata on the left quarter
    mask = (flags != 4).astype(np.uint8)
    values = np.where(mask == 0, chla, np.nan)
    ds = xr.Dataset({'Chla_OC2nasa': (('y', 'x'), values),
                     'TURB_dogliotti': (('y', 'x'), 2 * values),
                     'flags': (('y', 'x'), flags, {'flag_names': FLAG_NAMES,
                                                   'flag_descriptions': ['no data', 'cloud', 'water']}),
                     'mask': (('y', 'x'), mask)},
                    coords={'x': x, 'y': y, 'time': pd.Timestamp(time),
                            'forecast_period': forecast_period})
    return ds.rio.write_crs('EPSG:32631')


@pytest.fixture
def l2b_files(tmp_path):
    """Three L2B NetCDF files in the tree <tile>/YYYY/MM/DD/<product>; the CAMS forecast_period is a
    float in some products and a timedelta in others, the last image has no valid pixel."""
    specs = [('20250103T103339', 1.0, 4.32), ('20250108T103421', 2.0, np.timedelta64(4, 'h')),
             ('20250113T103309', np.nan, 4.32)]
    files = []
    for stamp, chla, forecast_period in specs:
        time = pd.Timestamp(stamp)
        name = f'S2A_MSIL2B_{stamp}_N0511_R108_T31TGJ_{stamp[:8]}T150000_V3.1.0'
        folder = tmp_path / '31TGJ' / f'{time:%Y}' / f'{time:%m}' / f'{time:%d}'
        folder.mkdir(parents=True)
        path = folder / f'{name}.nc'
        l2b_product(time, chla, forecast_period).to_netcdf(path)
        files.append(str(path))
    return files
