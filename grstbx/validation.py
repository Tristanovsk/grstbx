"""
Validation of the GRS L2A remote sensing reflectance with AERONET-OC, on the Sentinel-2 L1C subsets around the
AERONET-OC stations (static STAC catalogue of the bucket ``s3://aeronet-oc``, made by
``aeronet_oc_s2_subsets.py`` of the tutorial *Validation with AERONET-OC*).

The chain:

1. :func:`read_items`: the subsets of the catalogue (STAC Items), with the station and the AERONET-OC
   measurements around the overpass;
2. :func:`process_subsets`: GRS on the subsets, one process per image (``grstbx-validation grs``), L2A
   products in the tree of the bucket (``sentinel-2/l2a/<station>/YYYY/MM/DD/<name>/``);
3. :func:`extract_l2a`: statistics of the L2A pixels in a window around the station (ring around the
   platform by default, as the validation of GRS);
4. :func:`read_aeronet_oc`: AERONET-OC measurements of the in situ validation database (harmonised NetCDF:
   ``Rrs`` = Lwn / F0 directional as measured, ``Rrs_fQ`` and ``Rrs_IOP`` normalised);
5. :func:`matchup_table`: one row per image and Sentinel-2 band, paired with the nearest AERONET-OC
   measurement in time (Level 2.0 first) and band (within 15 nm); the quality criteria are kept as columns
   (:func:`quality`), so that they can be changed afterwards, e.g. in the dashboard;
6. :func:`metrics_table`: agreement statistics (:func:`grstbx.matchup.matchup_metrics`) by band, station...;
7. :func:`dashboard`: interactive analysis of the matchups (panel, holoviews, bokeh), in a notebook or as a web
   application (``grstbx-validation dashboard matchups.csv``).

Example
-------
>>> from grstbx import validation as val
>>> items = val.read_items('/data/satellite/validation/stac', stations=['AAOT'])
>>> runs = val.process_subsets(items, workers=2)
>>> table = val.matchup_table(items)
>>> val.metrics_table(table[val.quality(table)], by='wl')
"""

import argparse
import contextlib
import datetime
import gc
import tempfile
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .driver import to_float32
from .matchup import matchup_metrics

#: harmonised AERONET-OC dataset of the in situ validation database (``GRSTBX_AERONET_OC`` to change it)
AERONET_OC = os.environ.get('GRSTBX_AERONET_OC', '/data/insitu/validation_db/harmonised/aeronet_oc.nc')
#: harmonised HYPERNETS dataset of the in situ validation database (``GRSTBX_HYPERNETS`` to change it)
HYPERNETS = os.environ.get('GRSTBX_HYPERNETS', '/data/insitu/validation_db/harmonised/hypernets.nc')
#: content of the bucket of the subsets: catalogue ``stac/``, subsets ``sentinel-2/l1c``, L2A ``sentinel-2/l2a``
DATA_ROOT = os.environ.get('GRSTBX_VALIDATION_ROOT', '/data/satellite/validation')
#: URL of the bucket in the assets of the Items -> local folder
BUCKET_URL = 's3://aeronet-oc/'
#: CAMS files: daily ``YYYY/MM/DD/YYYY-MM-DD-cams-global-atmospheric-composition-forecasts.nc``, else monthly
CAMS_DIR = os.environ.get('GRSTBX_CAMS_DIR', '/data/cams/world')
#: in situ Rrs variables: directional (as measured), f/Q (Morel et al. 2002), IOP (Lee et al. 2011)
INSITU_VARIABLES = ('Rrs', 'Rrs_fQ', 'Rrs_IOP')
#: Sentinel-2 bands of the matchups (nm, ``wl`` coordinate of the GRS L2A products)
S2_BANDS = (443, 490, 560, 665, 705, 740, 783, 842, 865)
#: GRS stops with ``ValueError: zero-size array to reduction operation fmin which has no identity`` on some images
#: without clear water pixel (all cloudy or outside the swath) instead of flagging them ``no_water``
FMIN_ERROR = ('fmin', 'no identity')
#: default quality criteria of the matchups (see :func:`quality`)
QC = {'max_lag_min': 60, 'min_valid': 0.5, 'max_cv': 0.2, 'levels': ('lev20', 'lev15')}


# ------------------------------------------------------------------------------------------------------------
# subsets of the catalogue
# ------------------------------------------------------------------------------------------------------------

def _local(href, root=DATA_ROOT, url=BUCKET_URL):
    """Local path of an asset of the bucket."""
    return Path(root) / href[len(url):] if href.startswith(url) else Path(href)


def read_items(catalog=None, stations=None, start=None, end=None, root=DATA_ROOT):
    """
    Subsets of the static STAC catalogue, one row per image.

    :param catalog: folder of the catalogue, its ``catalog.json`` or ``items.json`` (default:
                    ``<root>/stac``)
    :param stations: stations to keep (``aeronet_oc:station``), None: all
    :param start: first date (included), None: all
    :param end: last date (included), None: all
    :param root: local folder of the bucket, to resolve the URLs of the assets
    :return: DataFrame: id, time, station, sites, lon, lat, n_obs, n_obs_lev20, dt_min, tile, platform,
             cloud_cover, subset (local path), item (properties)
    """
    catalog = Path(catalog or Path(root) / 'stac')
    if catalog.is_dir():
        catalog = catalog / 'catalog.json'
    if catalog.name == 'catalog.json':
        catalog = catalog.parent / 's2-l1c-aeronet-oc' / 'items.json'
    items = json.loads(catalog.read_text())['features']
    rows = []
    for item in items:
        p = item['properties']
        rows.append({'id': item['id'], 'time': pd.Timestamp(p['datetime']).tz_localize(None),
                     'station': p.get('aeronet_oc:station'), 'sites': p.get('aeronet_oc:sites'),
                     'lon': p.get('aeronet_oc:lon'), 'lat': p.get('aeronet_oc:lat'),
                     'n_obs': p.get('aeronet_oc:n_obs'), 'n_obs_lev20': p.get('aeronet_oc:n_obs_lev20'),
                     'dt_min': p.get('aeronet_oc:dt_min'), 'tile': p.get('grid:code', '').removeprefix('MGRS-'),
                     'platform': p.get('platform'), 'cloud_cover': p.get('eo:cloud_cover'),
                     'subset': str(_local(item['assets']['data']['href'], root)), 'item': p})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if stations is not None:
        table = table[table.station.isin([stations] if isinstance(stations, str) else stations)]
    if start is not None:
        table = table[table.time >= pd.Timestamp(start)]
    if end is not None:
        table = table[table.time < pd.Timestamp(end) + pd.Timedelta('1D')]
    return table.sort_values(['station', 'time']).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------
# GRS on the subsets
# ------------------------------------------------------------------------------------------------------------

def l2a_path(subset, version, root=DATA_ROOT):
    """
    L2A product of a subset: ``<root>/sentinel-2/l2a/<station>/YYYY/MM/DD/<name>/<name>.nc``, ``<name>`` the name
    of the subset with ``MSIL1C`` replaced by ``MSIL2AGRS`` and ``_V<version>`` appended.
    """
    subset = Path(subset)
    name = subset.stem.replace('_MSIL1C_', '_MSIL2AGRS_') + f'_V{version}'
    station, year, month, day = subset.parts[-5:-1]
    return Path(root) / 'sentinel-2' / 'l2a' / station / year / month / day / name / f'{name}.nc'


def l2a_state(path):
    """``done`` (product with its STAC Item), ``no_water``, ``failed`` or ``to_run``."""
    stem = str(path)[:-3]
    if Path(stem + '.json').exists():
        return 'done'
    if Path(stem + '_no_water.txt').exists():
        return 'no_water'
    if Path(stem + '.err').exists():
        return 'failed'
    return 'to_run'


def cams_file(time, cams_dir=CAMS_DIR):
    """CAMS file of the day if it exists, else of the month; None if none."""
    time = pd.Timestamp(time)
    for path in [Path(cams_dir) / f'{time:%Y/%m/%d}' / f'{time:%Y-%m-%d}-cams-global-atmospheric-composition-forecasts.nc',
                 Path(cams_dir) / f'cams_forecast_{time:%Y-%m}.nc']:
        if path.exists():
            return path
    return None


def run_grs(subset, cams, out, options=None):
    """
    GRS on one subset, in this process: L2A product ``out`` (NetCDF, with its ancillary file and the
    quicklooks of GRS) and its STAC Item (:func:`grstbx.batch.write_l2a_item`), written last. An image without
    enough water gets the marker ``<name>_no_water.txt``, also when GRS stops on it with the ``fmin`` error
    (:data:`FMIN_ERROR`, no clear water pixel); another failure writes ``<name>.err`` and is raised again.

    :return: ``'done'`` or ``'no_water'``
    """
    import grs
    from .batch import write_l2a_item

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out)[:-3]
    try:
        process = grs.Process()
        process.execute(str(subset), cams_file=str(cams), resolution=20, **(options or {}))
        if process.no_water:
            Path(stem + '_no_water.txt').write_text(json.dumps(process.no_water_info, indent=1, default=str))
            return 'no_water'
        process.odir = str(out)
        process.write_output()
        write_l2a_item(str(out))
        Path(stem + '.err').unlink(missing_ok=True)
        return 'done'
    except Exception as exc:
        if isinstance(exc, ValueError) and all(k in str(exc) for k in FMIN_ERROR):
            Path(stem + '_no_water.txt').write_text(json.dumps(
                {'image': str(subset), 'reason': 'no clear water pixel: GRS stopped with the fmin error',
                 'error': f'{type(exc).__name__}: {exc}'}, indent=1))
            Path(stem + '.err').unlink(missing_ok=True)
            return 'no_water'
        Path(stem + '.err').write_text(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
        raise


def process_subsets(items, version=None, root=DATA_ROOT, cams_dir=CAMS_DIR, workers=2, options=None,
                    retry_failed=False, python=None, verbose=True, images_per_process=20):
    """
    GRS on the subsets of a table of :func:`read_items` that have no L2A product yet, ``workers`` processes at a
    time, each processing ``images_per_process`` images one after the other (``grstbx-validation grs-batch``):
    grs is imported once per process instead of once per image (a subset takes a few seconds to process, about
    as long as the imports), and the memory is released at the end of each process. The images left by a process
    that died (e.g. killed when the machine runs out of memory) are run again one per process
    (``grstbx-validation grs-one``), so that a crash only fails its image. The images without CAMS file are
    skipped.

    :param items: table of :func:`read_items`
    :param version: GRS version of the L2A names (default: the version of the installed grs)
    :param options: options of ``grs.Process.execute`` (dict), e.g. ``{'glint_aot': True}``
    :param retry_failed: run again the images that failed (``.err`` file)
    :param python: Python interpreter of the processes (default: this one); the environment (``PYTHONPATH``)
                   is passed on
    :param images_per_process: images processed by each process (1: one process per image)
    :return: table: subset, l2a, cams, state (``done``, ``no_water``, ``failed``, ``no_cams``), seconds
    """
    if version is None:
        import grs
        version = grs.__version__
    python = python or sys.executable
    rows = []
    for row in items.itertuples():
        out = l2a_path(row.subset, version, root)
        cams = cams_file(row.time, cams_dir)
        state = l2a_state(out)
        if state == 'failed' and retry_failed:
            state = 'to_run'
        if state == 'to_run' and cams is None:
            state = 'no_cams'
        rows.append({'subset': row.subset, 'l2a': str(out), 'cams': str(cams) if cams else '', 'state': state,
                     'seconds': np.nan})
    runs = pd.DataFrame(rows)
    todo = runs.index[runs.state == 'to_run']
    if verbose:
        print(f'{len(runs)} subsets: {runs.state.value_counts().to_dict()}', flush=True)

    def one(i):
        row = runs.loc[i]
        t0 = time.time()
        cmd = [python, '-m', 'grstbx.validation', 'grs-one', row.subset, row.cams, row.l2a,
               '--options', json.dumps(options or {})]
        result = subprocess.run(cmd, capture_output=True, text=True)
        state = l2a_state(row.l2a)
        if result.returncode and state == 'to_run':          # killed (e.g. memory): keep the trace
            Path(row.l2a).parent.mkdir(parents=True, exist_ok=True)
            Path(str(row.l2a)[:-3] + '.err').write_text(f'status {result.returncode}\n{result.stderr[-3000:]}')
            state = 'failed'
        return [(i, state, time.time() - t0)]

    def batch(indexes):
        """Images of one process (grs-batch), then the ones it left (died) one per process."""
        if len(indexes) == 1:
            return one(indexes[0])
        jobs = [{'subset': runs.loc[i, 'subset'], 'cams': runs.loc[i, 'cams'], 'out': runs.loc[i, 'l2a']}
                for i in indexes]
        fd, jobs_file = tempfile.mkstemp(prefix='grstbx_grs_batch_', suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(jobs, f)
        try:
            result = subprocess.run([python, '-m', 'grstbx.validation', 'grs-batch', jobs_file,
                                     '--options', json.dumps(options or {})], capture_output=True, text=True)
        finally:
            os.remove(jobs_file)
        reported = {}
        for line in result.stdout.splitlines():
            if line.startswith('{"l2a"'):
                with contextlib.suppress(ValueError):
                    record = json.loads(line)
                    reported[record['l2a']] = (record['state'], record['seconds'])
        out = []
        for i in indexes:
            if runs.loc[i, 'l2a'] in reported:
                out.append((i, *reported[runs.loc[i, 'l2a']]))
            else:                            # not reached, or being processed when the process died
                out.extend(one(i))
        return out

    size = max(1, int(images_per_process))
    batches = [list(todo[k:k + size]) for k in range(0, len(todo), size)]
    k = 0
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for results in pool.map(batch, batches):
            for i, state, seconds in results:
                k += 1
                runs.loc[i, ['state', 'seconds']] = state, seconds
                if verbose:
                    print(f'[{k}/{len(todo)}] {state:8s} {Path(runs.loc[i, "l2a"]).name} ({seconds:.0f} s)',
                          flush=True)
    return runs


# ------------------------------------------------------------------------------------------------------------
# extraction around the station
# ------------------------------------------------------------------------------------------------------------

def window_mask(ny, nx, window='ring', size=(4, 7)):
    """
    Pixels of the window around the centre of the image (the station): ``'ring'``, at ``size[0]`` to ``size[1]``
    pixels (Chebyshev distance) from the centre, without the platform and its shadow; or ``'box'``, square of
    ``size`` x ``size`` pixels centred on the station.

    :return: boolean xarray.DataArray (y, x)
    """
    dy = xr.DataArray(np.abs(np.arange(ny) - ny // 2), dims='y')
    dx = xr.DataArray(np.abs(np.arange(nx) - nx // 2), dims='x')
    dist = np.maximum(dy, dx)
    if window == 'ring':
        return (dist >= size[0]) & (dist <= size[1])
    if window == 'box':
        return dist <= int(size) // 2
    raise ValueError(f"window must be 'ring' or 'box', not {window!r}")


def extract_window(ds, window='ring', size=(4, 7), platform_nir=8e-4, bands=S2_BANDS):
    """
    Statistics of the valid water pixels of an L2A product in the window around the station (centre of the
    subset): water pixels (``mask == 0``) with a finite Rrs in all the bands, without the pixels much brighter
    than the median of the window at 865 nm (platform, its shadow, boats: ``platform_nir`` sr-1 above).

    :param ds: GRS L2A product (dataset with Rrs, mask; aot550, BRDFg, wind and the angles if present)
    :return: dict: n_window, n_valid, frac_valid, cv560 (coefficient of variation of Rrs(560)), the medians of
             aot550_sat, BRDFg, wind_sat, sza, vza, raa, and for each band ``Rrs_sat_<wl>`` (median),
             ``Rrs_sat_mean_<wl>``, ``Rrs_sat_std_<wl>``
    """
    win = window_mask(ds.sizes['y'], ds.sizes['x'], window, size)
    nir = ds.Rrs.sel(wl=865, method='nearest').where(win)
    bright = (nir - nir.median()) > platform_nir
    valid = win & (ds['mask'] == 0) & ~bright & np.isfinite(ds.Rrs).all('wl')
    n = int(valid.sum())
    rrs = ds.Rrs.where(valid)
    r560 = rrs.sel(wl=560, method='nearest')
    out = {'n_window': int(win.sum()), 'n_valid': n, 'frac_valid': n / max(int(win.sum()), 1),
           'cv560': float(r560.std() / r560.mean()) if n > 1 else np.nan}
    for var, name in [('aot550', 'aot550_sat'), ('BRDFg', 'BRDFg'), ('wind', 'wind_sat'), ('sza', 'sza'),
                      ('vza', 'vza'), ('raa', 'raa')]:
        out[name] = float(ds[var].where(valid).median()) if var in ds and n else np.nan
    wls = [wl for wl in bands if wl in ds.wl.values]
    sel = rrs.sel(wl=wls)
    median, mean, std = sel.median(['y', 'x']).values, sel.mean(['y', 'x']).values, sel.std(['y', 'x']).values
    for k, wl in enumerate(wls):
        out[f'Rrs_sat_{wl}'] = float(median[k]) if n else np.nan
        out[f'Rrs_sat_mean_{wl}'] = float(mean[k]) if n else np.nan
        out[f'Rrs_sat_std_{wl}'] = float(std[k]) if n > 1 else np.nan
    return out


def aerosol_model(attrs):
    """
    Aerosol model of a GRS L2A product, from its attributes: the OPAC model selected by GRS (``opac_model``),
    or the mixture of the LUT models (``aerosol_combination``: JSON {model: weight, AOT fraction at 550 nm}),
    written ``'0.90 WASO_rh0 + 0.10 MACL_rh70'`` (largest weight first).

    :return: (aerosol_model, aerosol_combination): two strings, empty if unknown
    """
    mixture = attrs.get('aerosol_combination')
    if mixture:
        try:
            weights = {k: float(v) for k, v in json.loads(mixture).items() if float(v) > 0}
        except (TypeError, ValueError):
            weights = {}
        if len(weights) > 1:
            total = sum(weights.values())
            label = ' + '.join(f'{w / total:.2f} {m}' for m, w in sorted(weights.items(), key=lambda kv: -kv[1]))
            return label, json.dumps(weights)
        if weights:
            return next(iter(weights)), json.dumps(weights)
    model = str(attrs.get('opac_model') or '')
    return model, json.dumps({model: 1.0}) if model else ''


def add_aerosol_model(table, workers=8):
    """
    Add the columns ``aerosol_model`` and ``aerosol_combination`` (see :func:`aerosol_model`) to a matchup
    table made before they existed, from the attributes of its L2A products (no extraction).
    """
    def one(path):
        try:
            with xr.open_dataset(path) as ds:
                return path, *aerosol_model(ds.attrs)
        except Exception:
            return path, '', ''

    paths = table.l2a.dropna().unique()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        found = {p: (m, c) for p, m, c in pool.map(one, paths)}
    table = table.copy()
    table['aerosol_model'] = table.l2a.map(lambda p: found.get(p, ('', ''))[0])
    table['aerosol_combination'] = table.l2a.map(lambda p: found.get(p, ('', ''))[1])
    return table


def extract_l2a(paths, window='ring', size=(4, 7), platform_nir=8e-4, workers=4):
    """
    :func:`extract_window` for a list of L2A products (NetCDF files), read in parallel.

    :return: DataFrame, one row per product: l2a, version, aerosol_model, aerosol_combination (see
             :func:`aerosol_model`) and the statistics
    """
    def one(path):
        try:
            with xr.open_dataset(path) as ds:
                version = str(ds.attrs.get('version', ''))
                model, mixture = aerosol_model(ds.attrs)
                ds = ds[[v for v in ('Rrs', 'mask', 'aot550', 'BRDFg', 'wind', 'sza', 'vza', 'raa') if v in ds]]
                ds = to_float32(ds).load()          # half the memory of the float64 decoding
            return {'l2a': str(path), 'version': version, 'aerosol_model': model, 'aerosol_combination': mixture,
                    **extract_window(ds, window, size, platform_nir)}
        except Exception as exc:
            return {'l2a': str(path), 'error': f'{type(exc).__name__}: {exc}'}

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return pd.DataFrame(list(pool.map(one, [str(p) for p in paths])))


# ------------------------------------------------------------------------------------------------------------
# in situ data
# ------------------------------------------------------------------------------------------------------------

def _aot550(df):
    """AOT at 550 nm, Angstrom law between the AOT around 490 and 667 nm (or the nearest available bands)."""
    def first(*names):
        for name in names:
            if name in df and df[name].notna().any():
                return df[name], float(name.split('_')[-1])
        return None, None

    a1, w1 = first('src_aot_490', 'src_aot_500', 'src_aot_440', 'src_aot_443')
    a2, w2 = first('src_aot_667', 'src_aot_675', 'src_aot_681', 'src_aot_620', 'src_aot_865', 'src_aot_870')
    if a1 is None or a2 is None:
        return pd.Series(np.nan, index=df.index)
    alpha = -np.log(a2 / a1) / np.log(w2 / w1)
    return a1 * (550 / w1) ** -alpha


def _insitu_variables(path, variables, prefixes=()):
    """
    Variables of an in situ dataset of the validation database (one ``obs`` dimension), loaded in memory once per
    process and file version: the other variables are not read (the text variables of the harmonised AERONET-OC
    file took 50 s to decode at each opening), and the stations of a matchup table are taken from memory.

    :param variables: names of the variables (absent ones ignored)
    :param prefixes: prefixes of other variables to load (e.g. ``'src_aot_'``)
    :return: xarray.Dataset (do not modify it: it is shared by the calls)
    """
    path = os.path.abspath(str(path))
    return _load_insitu(path, os.path.getmtime(path), tuple(variables), tuple(prefixes))


@lru_cache(maxsize=4)
def _load_insitu(path, mtime, variables, prefixes):
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        names = list(nc.variables)
    keep = set(variables) | {v for v in names if v.startswith(prefixes)} if prefixes else set(variables)
    with xr.open_dataset(path, drop_variables=[v for v in names if v not in keep]) as ds:
        return ds.load()


def read_aeronet_oc(sites, path=AERONET_OC, levels=(2.0, 1.5), start=None, end=None):
    """
    AERONET-OC measurements of sites from the harmonised dataset of the in situ validation database.

    :param sites: AERONET-OC sites (``subsource``), e.g. ``['Venise', 'AAOT']``
    :param levels: quality levels kept (2.0, 1.5)
    :return: DataFrame indexed by time: site, level (``lev20``, ``lev15``), sza_insitu, vza_insitu, raa_insitu,
             wind_insitu,
             aot550_insitu, and ``<var>_<wl>`` for the Rrs variables of :data:`INSITU_VARIABLES` and the
             wavelengths with data (nominal wavelength, nm)
    """
    sites = [sites] if isinstance(sites, str) else list(sites)
    ds = _insitu_variables(path, ('time', 'subsource', 'qc_level', 'sza', 'vza', 'raa', 'wind_speed', 'wl')
                           + INSITU_VARIABLES, prefixes=('src_aot_',))
    keep = np.isin(ds.subsource.values, sites) & np.isin(ds.qc_level.values, levels)
    if start is not None:
        keep &= ds.time.values >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        keep &= ds.time.values <= np.datetime64(pd.Timestamp(end))
    sub = ds.isel(obs=np.flatnonzero(keep))
    scalars = ['time', 'subsource', 'qc_level', 'sza', 'vza', 'raa', 'wind_speed'] + \
              [v for v in sub.data_vars if v.startswith('src_aot_')]
    df = sub[[v for v in scalars if v in sub]].to_dataframe().reset_index(drop=True)
    df['aot550_insitu'] = _aot550(df)
    df = df.rename(columns={'subsource': 'site', 'wind_speed': 'wind_insitu', 'sza': 'sza_insitu',
                            'vza': 'vza_insitu', 'raa': 'raa_insitu'})
    df['level'] = np.where(df.pop('qc_level') == 2, 'lev20', 'lev15')
    df = df.drop(columns=[c for c in df if c.startswith('src_aot_')])
    for var in INSITU_VARIABLES:
        if var not in sub:
            continue
        values = sub[var].values
        for k, wl in enumerate(sub.wl.values):
            if np.isfinite(values[:, k]).any():
                df[f'{var}_{int(round(float(wl)))}'] = values[:, k]
    return df.set_index('time').sort_index()


def band_srf(srf, platform=None):
    """
    Spectral response functions of the Sentinel-2 bands, normalised to a unit integral on a 1 nm grid.

    :param srf: DataArray (wl, wl_hr) of the SRF (variable ``SRF`` of the GRSdriver L1C products and subsets), or
                the path of such a product
    :return: DataArray (wl, wl_hr)
    """
    if not isinstance(srf, xr.DataArray):
        with xr.open_dataset(srf) as ds:
            srf = ds['SRF'].load()
    srf = srf.fillna(0).clip(min=0)
    return srf / srf.sum('wl_hr')


def convolve_srf(rrs, srf, bands=S2_BANDS):
    """
    Hyperspectral Rrs convolved with the spectral response functions of the Sentinel-2 bands.

    :param rrs: DataArray (..., wl) on a regular grid (nm)
    :param srf: SRF of :func:`band_srf`
    :return: DataArray (..., wl) at the nominal wavelengths of ``bands``; NaN where the in situ spectrum does not
             cover 99 % of the response of the band
    """
    srf = srf.sel(wl=[wl for wl in bands if wl in srf.wl.values])
    srf = srf.rename(wl='band', wl_hr='wl').interp(wl=rrs.wl.values).fillna(0)
    srf = srf / srf.sum('wl')
    valid = rrs.notnull()
    out = (rrs.fillna(0) * srf).sum('wl') / (valid * srf).sum('wl')
    coverage = (valid * srf).sum('wl')
    return out.where(coverage >= 0.99).rename(band='wl')


def read_hypernets(sites, path=HYPERNETS, srf=None, start=None, end=None, qc_flags=(0,), dir_norm=None,
                   bands=S2_BANDS):
    """
    HYPERNETS (WATERHYPERNET) measurements of sites from the harmonised dataset of the in situ validation database,
    convolved with the spectral response functions of the Sentinel-2 bands, in the format of
    :func:`read_aeronet_oc` (to be used as the ``reader`` of :func:`matchup_table`).

    :param sites: HYPERNETS sites (first part of ``subsource``), e.g. ``['BEFR', 'MAFR']``
    :param srf: SRF of :func:`band_srf` (or the path of an L1C product or subset); one SRF for all the images
                (the bands of S2A, S2B and S2C differ by less than 1 nm, except 783 nm)
    :param qc_flags: values of ``qc_flag`` kept (0: QC-approved products)
    :param dir_norm: optional factors (DataArray (sza, wl)) converting the measured Rrs to nadir viewing;
                     adds ``Rrs_nadir_<wl>``
    :return: DataFrame indexed by time: site, level (``qc<qc_flag>``), sza_insitu, vza_insitu, raa_insitu,
             wind_insitu, aot550_insitu (NaN), ``Rrs_<wl>`` (as measured) and ``Rrs_nadir_<wl>`` at the nominal
             wavelengths of the Sentinel-2 bands
    """
    if srf is None:
        raise ValueError('srf: SRF of the Sentinel-2 bands (band_srf) needed for the hyperspectral in situ data')
    if not isinstance(srf, xr.DataArray) or 'wl_hr' not in srf.dims or abs(float(srf.sum('wl_hr')[0]) - 1) > 1e-6:
        srf = band_srf(srf)
    sites = [sites] if isinstance(sites, str) else list(sites)
    ds = _insitu_variables(path, ('time', 'subsource', 'sza', 'vza', 'raa', 'wind_speed', 'qc_flag', 'Rrs', 'wl'))
    site = np.array([str(s).split('_')[0] for s in ds.subsource.values])
    keep = np.isin(site, sites) & np.isin(ds.qc_flag.values, qc_flags)
    if start is not None:
        keep &= ds.time.values >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        keep &= ds.time.values <= np.datetime64(pd.Timestamp(end))
    idx = np.flatnonzero(keep)
    sub = ds[['time', 'sza', 'vza', 'raa', 'wind_speed', 'qc_flag', 'Rrs']].isel(obs=idx)
    df = pd.DataFrame({'time': sub.time.values, 'site': site[idx],
                       'level': [f'qc{q}' for q in sub.qc_flag.values],
                       'sza_insitu': sub.sza.values, 'vza_insitu': sub.vza.values, 'raa_insitu': sub.raa.values,
                       'wind_insitu': sub.wind_speed.values, 'aot550_insitu': np.nan})
    variables = {'Rrs': sub.Rrs}
    if dir_norm is not None:
        factor = dir_norm.interp(wl=sub.wl.values).interp(sza=sub.sza)
        variables['Rrs_nadir'] = factor * sub.Rrs
    for var, rrs in variables.items():
        conv = convolve_srf(rrs, srf, bands).transpose('obs', 'wl')
        for k, wl in enumerate(conv.wl.values):
            df[f'{var}_{int(wl)}'] = conv.values[:, k]
    return df.set_index('time').sort_index()


def _nearest(insitu, times, max_lag):
    """Nearest in situ record of each time within max_lag (NaN row if none), Level 2.0 first."""
    out = pd.DataFrame(index=range(len(times)), columns=['time_insitu'] + list(insitu.columns))
    times = pd.DatetimeIndex(times)
    # records at the same time (sites of one station, e.g. AAOT and Venise): one kept, Level 2.0 first
    order = np.lexsort((insitu.level.ne('lev20').values, insitu.index.values))
    insitu = insitu.iloc[order]
    insitu = insitu[~insitu.index.duplicated(keep='first')]
    for subset in ([insitu[insitu.level == 'lev20'], insitu] if (insitu.level == 'lev20').any() else [insitu]):
        if subset.empty:
            continue
        pos = subset.index.get_indexer(times, method='nearest')
        found = subset.iloc[pos].reset_index().rename(columns={'time': 'time_insitu'})
        lag = (found.time_insitu - times).abs()
        use = (lag <= pd.Timedelta(max_lag)).values & out.time_insitu.isna().values
        out.loc[use] = found.loc[use, out.columns].values
    out['time_insitu'] = pd.to_datetime(out.time_insitu)
    return out


# ------------------------------------------------------------------------------------------------------------
# matchups
# ------------------------------------------------------------------------------------------------------------

def band_pairs(insitu_wl, bands=S2_BANDS, tolerance=15):
    """Sentinel-2 band -> nearest in situ wavelength within ``tolerance`` nm (bands without one left out)."""
    insitu_wl = sorted(insitu_wl)
    pairs = {}
    for wl in bands:
        if insitu_wl:
            near = min(insitu_wl, key=lambda w: abs(w - wl))
            if abs(near - wl) <= tolerance:
                pairs[wl] = near
    return pairs


def matchup_table(items, extractions=None, version=None, root=DATA_ROOT, insitu_path=AERONET_OC, max_lag='3h',
                  tolerance=15, window='ring', size=(4, 7), platform_nir=8e-4, workers=4, reader=None,
                  verbose=True):
    """
    Matchup table: one row per image and Sentinel-2 band, with the satellite statistics of the window around the
    station and the nearest AERONET-OC measurement within ``max_lag`` (the nearest Level 2.0 one if any, else
    the nearest Level 1.5 one), at the nearest AERONET-OC band within ``tolerance`` nm. All the images with an L2A
    product are kept: the quality criteria are columns (lag_min, frac_valid, cv560, level...), applied by
    :func:`quality`.

    :param items: table of :func:`read_items`
    :param extractions: table of :func:`extract_l2a` (default: extracted from the L2A products of the items)
    :param version: GRS version of the L2A products (default: the installed grs)
    :param reader: in situ data of the sites of a station, ``reader(sites, start=..., end=...)`` -> DataFrame in
                   the format of :func:`read_aeronet_oc` (``Rrs_<wl>`` and the normalised ``Rrs_<norm>_<wl>``,
                   in the columns ``Rrs_insitu`` and ``Rrs_insitu_<norm>`` of the table); default: AERONET-OC of
                   ``insitu_path``, e.g. ``functools.partial(read_hypernets, srf=...)`` for HYPERNETS
    :return: DataFrame (long format): station, image, l2a, time, platform, tile, cloud_cover, version, the
             extraction statistics, aerosol_model, aerosol_combination, site, level, time_insitu, lag_min, wl,
             wl_insitu, Rrs_sat, Rrs_sat_mean,
             Rrs_sat_std, Rrs_insitu (directional), Rrs_insitu_fQ, Rrs_insitu_IOP, residual (sat - in situ),
             lon, lat
    """
    if version is None and extractions is None:
        import grs
        version = grs.__version__
    items = items.copy()
    if extractions is None:
        items['l2a'] = [str(l2a_path(s, version, root)) for s in items.subset]
        items = items[[l2a_state(p) == 'done' for p in items.l2a]]
        if verbose:
            print(f'{len(items)} L2A products to extract', flush=True)
        extractions = extract_l2a(items.l2a, window, size, platform_nir, workers)
    else:
        items['l2a'] = [str(l2a_path(s, version or '', root)) for s in items.subset]
    if extractions.empty or 'l2a' not in extractions:
        if verbose:
            print('no L2A product to extract: run process_subsets first', flush=True)
        return pd.DataFrame()
    sat = items.drop(columns=['item'], errors='ignore').merge(extractions, on='l2a', how='inner')
    if 'error' in sat:
        sat = sat[sat['error'].isna()].drop(columns='error')
    rows = []
    for station, group in sat.groupby('station'):
        sites = group.sites.iloc[0]
        period = dict(start=group.time.min() - pd.Timedelta(max_lag), end=group.time.max() + pd.Timedelta(max_lag))
        insitu = reader(sites, **period) if reader else read_aeronet_oc(sites, insitu_path, **period)
        if insitu.empty:
            continue
        near = _nearest(insitu, group.time, max_lag)
        near.index = group.index
        merged = pd.concat([group, near], axis=1)
        merged = merged[merged.time_insitu.notna()]
        wl_insitu = sorted({int(c.split('_')[-1]) for c in insitu if re.fullmatch(r'Rrs_\d+', c)})
        # in situ Rrs: as measured (Rrs_<wl>) and normalised (Rrs_<norm>_<wl>) -> Rrs_insitu, Rrs_insitu_<norm>
        norms = sorted({m.group(1) for c in insitu if (m := re.fullmatch(r'Rrs_([A-Za-z]\w*?)_\d+', c))})
        columns = [('Rrs', 'Rrs_insitu')] + [(f'Rrs_{n}', f'Rrs_insitu_{n}') for n in norms]
        for wl in band_pairs(wl_insitu, tolerance=tolerance):
            if f'Rrs_sat_{wl}' not in merged:
                continue
            part = merged[[c for c in merged if not re.match(r'Rrs(_[A-Za-z]\w*?)?_\d+$', c)]].copy()
            part['wl'] = wl
            part['Rrs_sat'] = merged[f'Rrs_sat_{wl}']
            part['Rrs_sat_mean'] = merged[f'Rrs_sat_mean_{wl}']
            part['Rrs_sat_std'] = merged[f'Rrs_sat_std_{wl}']
            # per record, the nearest in situ band with a value (the sites of a station, or one site over time,
            # may have different bands: e.g. 560 nm at AAOT, 551 or 555 nm at Venise)
            part['wl_insitu'] = np.nan
            for _, col in columns:
                part[col] = np.nan
            for wl_in in sorted((w for w in wl_insitu if abs(w - wl) <= tolerance), key=lambda w: abs(w - wl)):
                values = pd.to_numeric(merged[f'Rrs_{wl_in}'], errors='coerce')
                use = part.wl_insitu.isna() & values.notna()
                part.loc[use, 'wl_insitu'] = wl_in
                for var, col in columns:
                    if f'{var}_{wl_in}' in merged:
                        part.loc[use, col] = pd.to_numeric(merged.loc[use, f'{var}_{wl_in}'], errors='coerce')
            part = part[part.wl_insitu.notna()].astype({'wl_insitu': int})
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table['image'] = [Path(p).stem for p in table.l2a]
    table['lag_min'] = (table.time_insitu - table.time).dt.total_seconds() / 60
    table['residual'] = table.Rrs_sat - table.Rrs_insitu
    for col in ['aot550_insitu', 'wind_insitu', 'sza_insitu', 'vza_insitu', 'raa_insitu']:
        if col in table:
            table[col] = pd.to_numeric(table[col], errors='coerce')
    table = table.drop(columns=['subset'], errors='ignore')
    first = ['station', 'image', 'time', 'wl', 'wl_insitu', 'Rrs_sat', 'Rrs_sat_std', 'Rrs_insitu', 'residual',
             'level', 'site', 'time_insitu', 'lag_min', 'n_valid', 'frac_valid', 'cv560']
    return table[[c for c in first if c in table] + [c for c in table if c not in first]].sort_values(
        ['station', 'time', 'wl']).reset_index(drop=True)


def quality(table, max_lag_min=QC['max_lag_min'], min_valid=QC['min_valid'], max_cv=QC['max_cv'],
            levels=QC['levels'], max_aot=None, max_glint=None, min_n_valid=1):
    """
    Matchups meeting the quality criteria: in situ measurement within ``max_lag_min`` minutes (Zibordi et al.
    2009: 1 h) and of the given levels, at least ``min_valid`` of the window valid, coefficient of variation of
    Rrs(560) in the window below ``max_cv`` (Bailey and Werdell 2006), optionally AOT(550) of GRS below
    ``max_aot`` and sunglint (BRDFg) below ``max_glint``. ``levels=None``: all the in situ levels.

    :return: boolean Series aligned with the table
    """
    ok = ((table.lag_min.abs() <= max_lag_min) & (table.frac_valid >= min_valid) & (table.cv560 <= max_cv)
          & (table.level.isin(levels) if levels is not None else True) & (table.n_valid >= min_n_valid)
          & np.isfinite(table.Rrs_sat) & np.isfinite(table.Rrs_insitu))
    if max_aot is not None:
        ok &= table.aot550_sat <= max_aot
    if max_glint is not None:
        ok &= table.BRDFg <= max_glint
    return ok


def metrics_table(table, by='wl', x='Rrs_insitu', y='Rrs_sat', min_n=3):
    """
    Agreement statistics (:func:`grstbx.matchup.matchup_metrics`) of the matchups, overall and by group.

    :param table: matchup table (filtered, e.g. ``table[quality(table)]``)
    :param by: column(s) of the groups, e.g. ``'wl'``, ``'station'``, ``['station', 'wl']``; None: overall only
    :param min_n: groups with fewer matchups left out
    :return: DataFrame indexed by the groups (first row: all the matchups, ``'all'``)
    """
    rows = {}
    if len(table) >= min_n:
        rows['all'] = matchup_metrics(table[x], table[y])
    if by is not None:
        for key, group in table.groupby(by):
            if len(group) >= min_n and group[[x, y]].notna().all(axis=1).sum() >= min_n:
                rows[key] = matchup_metrics(group[x], group[y])
    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index.name = by if isinstance(by, str) else ('/'.join(by) if by else None)
    return out


def save_table(table, path):
    """Matchup table to CSV or Parquet (from the extension), with the times in ISO format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    if 'sites' in table:
        table['sites'] = table.sites.map(lambda s: '+'.join(s) if isinstance(s, (list, tuple)) else s)
    if path.suffix == '.parquet':
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path


def read_table(path):
    """Matchup table written by :func:`save_table` (CSV or Parquet)."""
    path = str(path)
    table = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    for col in ['time', 'time_insitu']:
        if col in table:
            table[col] = pd.to_datetime(table[col])
    if 'version' in table:
        table['version'] = table.version.astype(str)
    return table


def dashboard(table, **kwargs):
    """Interactive dashboard of a matchup table (see :mod:`grstbx.validation_app`)."""
    from .validation_app import create_dashboard

    return create_dashboard(table, **kwargs)


# ------------------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------------------

def main(argv=None):
    """
    ``grstbx-validation``: validation of GRS with AERONET-OC on the subsets of the bucket ``aeronet-oc``.

    ::

        grstbx-validation grs [--stations AAOT ...] [--workers 2]     GRS on the subsets without L2A product
        grstbx-validation matchups -o matchups.csv [--stations ...]   matchup table
        grstbx-validation dashboard matchups.csv [--port 5007]        dashboard (web application)
        grstbx-validation grs-one <subset.nc> <cams.nc> <l2a.nc>      GRS on one subset (used by grs)
        grstbx-validation grs-batch <jobs.json>                     GRS on several subsets (used by grs)
    """
    ap = argparse.ArgumentParser(prog='grstbx-validation', description=main.__doc__.strip().split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    for name in ('grs', 'matchups'):
        p = sub.add_parser(name)
        p.add_argument('--root', default=DATA_ROOT, help='local folder of the bucket (catalogue in <root>/stac)')
        p.add_argument('--stations', nargs='*')
        p.add_argument('--start')
        p.add_argument('--end')
        p.add_argument('--workers', type=int, default=2)
    sub.choices['grs'].add_argument('--options', default='{}', help='options of grs.Process.execute (JSON)')
    sub.choices['grs'].add_argument('--retry-failed', action='store_true')
    sub.choices['grs'].add_argument('--images-per-process', type=int, default=20)
    sub.choices['matchups'].add_argument('-o', '--output', required=True, help='.csv or .parquet')
    sub.choices['matchups'].add_argument('--max-lag', default='3h')
    sub.choices['matchups'].add_argument('--insitu', default=AERONET_OC)
    p = sub.add_parser('dashboard')
    p.add_argument('tables', nargs='+', help='matchup tables (several: e.g. GRS versions, compared)')
    p.add_argument('--port', type=int, default=5007)
    p.add_argument('--address', default='localhost')
    p.add_argument('--show', action='store_true', help='open a browser')
    p = sub.add_parser('grs-one')
    p.add_argument('subset')
    p.add_argument('cams')
    p.add_argument('out')
    p.add_argument('--options', default='{}')
    p = sub.add_parser('grs-batch')
    p.add_argument('jobs', help='JSON list of {"subset", "cams", "out"}')
    p.add_argument('--options', default='{}')
    a = ap.parse_args(argv)

    if a.step in ('grs-one', 'grs-batch'):
        try:                         # first process killed if the machine runs out of memory
            Path('/proc/self/oom_score_adj').write_text('1000')
        except OSError:
            pass
    if a.step == 'grs-batch':
        # the images one after the other in this process; one JSON line per image on stdout (read by
        # process_subsets); the images already done (by a previous attempt) are skipped
        options = json.loads(a.options)
        for job in json.loads(Path(a.jobs).read_text()):
            state, t0 = l2a_state(job['out']), time.time()
            if state not in ('done', 'no_water'):
                try:
                    state = run_grs(job['subset'], job['cams'], job['out'], options)
                except Exception:
                    state = 'failed'         # .err file written by run_grs
            print(json.dumps({'l2a': job['out'], 'state': state, 'seconds': round(time.time() - t0, 1)}),
                  flush=True)
            gc.collect()                     # the products of the previous image released before the next one
        return 0
    if a.step == 'grs-one':
        try:
            state = run_grs(a.subset, a.cams, a.out, json.loads(a.options))
        except Exception as exc:
            print(f'{Path(a.out).name}: FAILED {type(exc).__name__}: {exc}')
            return 1
        print(f'{Path(a.out).name}: {state}')
        return 0
    if a.step == 'dashboard':
        import panel as pn
        table = pd.concat([read_table(path) for path in a.tables], ignore_index=True)
        pn.serve({'matchups': lambda: dashboard(table)}, port=a.port, address=a.address, show=a.show,
                 title='GRS / AERONET-OC matchups')
        return 0
    items = read_items(Path(a.root) / 'stac', stations=a.stations, start=a.start, end=a.end, root=a.root)
    if a.step == 'grs':
        runs = process_subsets(items, root=a.root, workers=a.workers, options=json.loads(a.options),
                               retry_failed=a.retry_failed, images_per_process=a.images_per_process)
        print(runs.state.value_counts().to_dict())
        return 0
    table = matchup_table(items, root=a.root, insitu_path=a.insitu, max_lag=a.max_lag, workers=a.workers)
    save_table(table, a.output)
    ok = quality(table)
    print(f'{table.image.nunique()} images, {ok.sum()} matchups (image x band) meeting the default criteria '
          f'-> {a.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
