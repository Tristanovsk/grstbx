"""
Surface pressure (``sp``) of the monthly CAMS global forecast files that lack it (2015-01 to 2016-05: the ADS
delivers only the mean sea level pressure ``msl`` for this period of the CAMS forecasts), derived from their
``msl`` and 2 m temperature ``t2m`` with the barometric formula and the effective orography of the CAMS model:

    sp = msl * exp(-g h / (Rd Tm)),   Tm = t2m + 0.5 * 0.0065 * h

The effective orography ``h`` (m) of each grid cell is the mean over the complete months (with ``sp``) of the same
formula inverted, h = Rd Tm / g * ln(msl / sp) (two fixed-point iterations for Tm). Over the sea h = 0 and sp = msl.

    python fill_cams_surface_pressure.py validate     # h from 2016-06..2016-08, sp of 2016-12 and 2017-03 compared
    python fill_cams_surface_pressure.py fill         # h from 2016-06..2017-05, sp added to the files without it
    python fill_cams_surface_pressure.py fill --months 2016-06    # missing times of a month with sp filled

``fill`` moves each original file to ``<cams_dir>/original_without_sp/`` and writes the file with ``sp``
(int16 packed as the other variables, attributes saying how it was derived) to its place, through a temporary file.
In a month with ``sp`` for part of its times (2016-06: ``sp`` delivered from 2016-06-21 09:00 only), the delivered
values are kept and only the missing times derived. GRS reads ``sp`` (``grs.cams``), the reason of this fix: it
stops on a subset whose time has no ``sp`` (``no valid value to compute the transmittance``).
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

CAMS_DIR = Path('/data/cams/world')
G, RD, GAMMA = 9.80665, 287.05, 0.0065


def months(first, last):
    return [str(m) for m in np.arange(np.datetime64(first, 'M'), np.datetime64(last, 'M') + 1)]


def path(month):
    return CAMS_DIR / f'cams_forecast_{month}.nc'


def orography(month_list, chunk=40):
    """Effective orography (m): mean over the times of the months of the inverted barometric formula."""
    total = count = None
    for month in month_list:
        with xr.open_dataset(path(month)) as ds:
            for i in range(0, ds.sizes['time'], chunk):
                part = ds[['msl', 'sp', 't2m']].isel(time=slice(i, i + chunk)).load().astype('float32')
                ratio = np.log(part.msl.values / part.sp.values)
                t2m = part.t2m.values
                h = RD * t2m / G * ratio
                for _ in range(2):
                    h = RD * (t2m + 0.5 * GAMMA * h) / G * ratio
                ok = np.isfinite(h)            # missing values of the files left out
                total = np.where(ok, h, 0).sum(0) if total is None else total + np.where(ok, h, 0).sum(0)
                count = ok.sum(0) if count is None else count + ok.sum(0)
        print(f'  {month}: {int(count.max())} times, {int((count == 0).sum())} cells without value', flush=True)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def derive_sp(msl, t2m, h):
    return msl * np.exp(-G * h / (RD * (t2m + 0.5 * GAMMA * h)))


def validate():
    h = orography(months('2016-06', '2016-08'))
    errors, errors_sea, errors_high = [], [], []
    for month in ['2016-12', '2017-03']:
        with xr.open_dataset(path(month)) as ds:
            part = ds[['msl', 'sp', 't2m']].isel(time=slice(0, None, 4)).load()
        err = (derive_sp(part.msl.values, part.t2m.values, h) - part.sp.values) / 100     # hPa
        errors.append(err.ravel())
        errors_sea.append(err[:, np.abs(h) < 10].ravel())
        errors_high.append(err[:, h > 1000].ravel())
    for name, e in [('all cells', errors), ('|h| < 10 m (sea, low coasts)', errors_sea), ('h > 1000 m', errors_high)]:
        e = np.concatenate(e)
        e = e[np.isfinite(e)]
        print(f'{name:30s} bias {np.mean(e):+.2f} hPa, std {np.std(e):.2f}, '
              f'|err| p95 {np.percentile(np.abs(e), 95):.2f}, max {np.abs(e).max():.1f} hPa (n={e.size})')


def _has_sp(month):
    try:
        with xr.open_dataset(path(month)) as ds:
            return 'sp' in ds.data_vars
    except (OSError, ValueError):          # unreadable file: rewritten from its original
        return False


def fill(only=None):
    """
    Add sp to the files without it, or to the months ``only`` (rewritten from their original in
    ``original_without_sp/``). The effective orography is cached in ``original_without_sp/cams_orography.nc``.
    """
    backup = CAMS_DIR / 'original_without_sp'
    backup.mkdir(exist_ok=True)
    missing = only or [m for m in months('2015-01', '2016-12')
                       if (path(m).exists() or (backup / path(m).name).exists()) and not _has_sp(m)]
    print('months to fill:', missing)
    if not missing:
        return
    cache = backup / 'cams_orography.nc'
    if cache.exists():
        h = xr.open_dataarray(cache).values
    else:
        h = orography(months('2016-06', '2017-05'))
        xr.DataArray(h, dims=('latitude', 'longitude'), name='h',
                     attrs={'units': 'm', 'long_name': 'effective orography of the CAMS model',
                            'comment': 'from msl, sp and t2m of the CAMS forecasts 2016-06 to 2017-05'}).to_netcdf(cache)
    for month in missing:
        src = backup / path(month).name
        if not src.exists():
            shutil.move(path(month), src)
        with xr.open_dataset(src) as ds:
            sp = derive_sp(ds.msl.load(), ds.t2m.load(), h).astype('float32')
            derived = 'derived from msl and t2m with the barometric formula and the effective orography of the CAMS ' \
                      'model estimated from the forecasts 2016-06 to 2017-05 (fill_cams_surface_pressure.py, grstbx)'
            if 'sp' in ds.data_vars:               # sp for part of the times: kept, the missing times derived
                given = ds.sp.load().astype('float32')
                missing_times = given.isnull().all(('latitude', 'longitude'))
                sp = given.fillna(sp)
                comment = (f'delivered by the ADS for {int((~missing_times).sum())} times; the '
                           f'{int(missing_times.sum())} times without it {derived}')
            else:
                comment = f'not delivered by the ADS for this period: {derived}'
            sp.attrs = {'units': 'Pa', 'long_name': 'Surface pressure', 'standard_name': 'surface_air_pressure',
                        'comment': comment}
            out = ds.assign(sp=sp)
            vmin, vmax = float(sp.min()), float(sp.max())
            scale = (vmax - vmin) / 65532          # packed values -32766..32766: -32767 is the fill value
            encoding = {'sp': {'dtype': 'int16', 'scale_factor': scale, 'add_offset': vmin + 32766 * scale,
                               '_FillValue': np.int16(-32767), 'missing_value': np.int16(-32767)}}
            tmp = path(month).with_name(f'.{path(month).name}.part')
            out.to_netcdf(tmp, encoding=encoding, format=ds.encoding.get('format', 'NETCDF4'))
        os.replace(tmp, path(month))
        with xr.open_dataset(path(month)) as check:
            assert 'sp' in check and np.isfinite(check.sp).all()
        print(f'{month}: sp added ({vmin / 100:.0f} to {vmax / 100:.0f} hPa), original in {backup}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('step', choices=['validate', 'fill'])
    ap.add_argument('--months', nargs='*', help='months to (re)write from their original, e.g. 2015-05')
    a = ap.parse_args()
    sys.exit(validate() if a.step == 'validate' else fill(a.months))
