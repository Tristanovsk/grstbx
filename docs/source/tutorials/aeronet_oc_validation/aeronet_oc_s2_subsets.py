"""
Sentinel-2 L1C subsets of 5 km x 5 km around the AERONET-OC stations, for the matchups of GRS L2A with the
AERONET-OC remote sensing reflectance, in a local S3 bucket with a static STAC catalogue.

Images: the Sentinel-2 overpasses (CDSE catalogue, last processing of each image) with at least one AERONET-OC
measurement (Level 1.5 or 2.0, validation database) within ``window`` hours (2 h, as radcalnet_oc
``s2_toa_matchup.py``). One tile per datatake: the tile whose data footprint covers the box best (the tiles of
the same datatake hold the same pixels); the station must be inside the footprint.

Subsets: product of GRSdriver (Sentinel2Driver, bands at 20 m, viewing angles computed per band and detector at
60 m and resampled, sun angles, SRF), box of 251 x 251 pixels at 20 m centred on the station (the station falls
in the central pixel), written as NetCDF with the bands and angles packed in int16 (as
``grs/notebook/validation/make_l1c_subsets.py``): about 1.3 MB per image, read directly by GRS
(``grs <file>.nc`` or ``grs.Process().execute(<file>.nc, ...)``, about 25 s per image).

The images already on this machine (``LOCAL_L1C``, same product name) are read in place; the others are
downloaded from the CDSE S3 interface (keys of ``~/.s3cfg``), subset and deleted one by one.

Layout: content of the bucket ``aeronet-oc`` of the local S3 service of grstbx (``grstbx-s3-bucket``) in
``/data/satellite/validation``, linked from ``/data/satellite/s3/aeronet-oc``::

    /data/satellite/validation/
        sentinel-2/l1c/<station>/YYYY/MM/DD/<name>.nc     subset (<name> = <Sentinel-2 name>_<station>)
        sentinel-2/l1c/<station>/YYYY/MM/DD/<name>.json   STAC Item, written last: the subset is complete
        sentinel-2/l1c/<station>/YYYY/MM/DD/<name>_ql.png quicklook (TOA RGB, station marked)
        sentinel-2/l2a/<station>/YYYY/MM/DD/<name>/       GRS L2A products of the subsets (run_grs_subset.py)
        stations.geojson                                  stations, sites and boxes
        stac/catalog.json                                 static STAC catalogue (one collection)
        stac/s2-l1c-aeronet-oc/collection.json, items.json, <id>/<id>.json
        work/plans/<station>.csv, work/logs/, work/download/   not in the bucket

Usage::

    python aeronet_oc_s2_subsets.py stations                       # stations of the validation database
    python aeronet_oc_s2_subsets.py plan [--stations AAOT ...]     # images to subset, plans/<station>.csv
    python aeronet_oc_s2_subsets.py run [--stations AAOT ...] [--workers 3]
    python aeronet_oc_s2_subsets.py catalog                        # STAC catalogue of the subsets

``run`` skips the subsets done (Item present) and the failures (``<name>.nc.err``, retried with
``--retry-failed``); it can be stopped and restarted at any time. One process per image (memory released).
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

AERONET_OC = '/data/insitu/validation_db/harmonised/aeronet_oc.nc'
DATA_ROOT = Path('/data/satellite/validation')        # content of the bucket
OUT = DATA_ROOT / 'sentinel-2' / 'l1c'               # subsets
L2A = DATA_ROOT / 'sentinel-2' / 'l2a'               # GRS L2A products of the subsets (run_grs_subset.py)
STAC_DIR = DATA_ROOT / 'stac'                        # STAC catalogue
WORK_ROOT = DATA_ROOT / 'work'                       # not in the bucket
PLANS = WORK_ROOT / 'plans'                          # images to subset, per station
WORK = WORK_ROOT / 'download'                        # full L1C images, deleted after subsetting
LOCAL_L1C = Path('/data/satellite/Sentinel-2/L1C')   # images already on this machine (never deleted)
BUCKET = 'aeronet-oc'
BUCKET_DIR = Path('/data/satellite/s3') / BUCKET
PREFIX = 'sentinel-2/l1c'
COLLECTION = 's2-l1c-aeronet-oc'

START = '2015-07-01'
HALF = 2500.          # half size of the box (m): 251 x 251 pixels at 20 m
RESOLUTION = 20
RESOLUTION_ANGLE = 60
WINDOW_H = 2.         # AERONET-OC measurement within this time of the overpass (h)

_NAME = re.compile(r'^(S2[A-D])_MSIL1C_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})')


# ------------------------------------------------------------------------------------------------------------
# stations
# ------------------------------------------------------------------------------------------------------------

def aeronet_obs(start=START):
    """AERONET-OC measurements of the validation database: site, time, lon, lat, level (1.5, 2.0)."""
    import xarray as xr

    with xr.open_dataset(AERONET_OC) as ds:
        obs = ds[['subsource', 'time', 'lat', 'lon', 'qc_level']].to_dataframe().reset_index(drop=True)
    obs = obs.rename(columns={'subsource': 'site', 'qc_level': 'level'})
    return obs[obs.time >= start]


def stations(obs=None):
    """
    Stations: the sites at the same position (e.g. Venise and AAOT, USC_SEAPRISM and USC_SEAPRISM_2) are one
    station, named after the site with the last measurement.

    :return: DataFrame indexed by station: lon, lat, sites (list), n_obs, first, last
    """
    obs = aeronet_obs() if obs is None else obs
    sites = obs.groupby('site').agg(lon=('lon', 'median'), lat=('lat', 'median'), n_obs=('time', 'size'),
                                    first=('time', 'min'), last=('time', 'max'))
    sites['key'] = [f'{lon:.3f},{lat:.3f}' for lon, lat in zip(sites.lon, sites.lat)]
    rows = []
    for _, group in sites.groupby('key'):
        group = group.sort_values('last')
        rows.append({'station': group.index[-1], 'lon': group.lon.iloc[-1], 'lat': group.lat.iloc[-1],
                     'sites': list(group.index), 'n_obs': int(group.n_obs.sum()),
                     'first': group['first'].min(), 'last': group['last'].max()})
    return pd.DataFrame(rows).set_index('station').sort_values('n_obs', ascending=False)


def utm_epsg(tile, lat):
    return (32600 if lat >= 0 else 32700) + int(tile[:2])


def station_box(lon, lat, epsg, half=HALF):
    """Box of the subset in the UTM projection of the tile: (xmin, ymin, xmax, ymax)."""
    from pyproj import Transformer

    x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    return x0 - half, y0 - half, x0 + half, y0 + half


def lonlat_box(lon, lat, half=HALF):
    """Box of the subset in lon/lat (local UTM zone), polygon."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    return transform(to_lonlat, box(*station_box(lon, lat, epsg, half)))


# ------------------------------------------------------------------------------------------------------------
# plan: images with AERONET-OC measurements around the overpass
# ------------------------------------------------------------------------------------------------------------

def subset_name(image, station):
    return f"{image.removesuffix('.SAFE')}_{station}"


def subset_path(image, station, time):
    time = pd.Timestamp(time)
    return OUT / station / f'{time:%Y/%m/%d}' / f'{subset_name(image, station)}.nc'


def plan_station(station, info, obs, end=None, window_h=WINDOW_H):
    """
    Images of a station: overpasses with AERONET-OC measurements within ``window_h``, one tile per datatake.

    :return: DataFrame, one row per image (columns of grstbx.download.search, and n_obs, n_obs_lev20,
             dt_min (signed, min, measurement - overpass), coverage of the box, path of the subset)
    """
    from shapely.geometry import Point, shape
    from grstbx import download

    end = end or pd.Timestamp.now().strftime('%Y-%m-%d')
    box = lonlat_box(info.lon, info.lat)
    times = obs[obs.site.isin(info.sites)][['time', 'level']].sort_values('time')
    if times.empty:
        return pd.DataFrame()
    start = max(pd.Timestamp(START), times.time.min().floor('D'))
    images = []
    for year in range(start.year, pd.Timestamp(end).year + 1):      # one request per year
        y0, y1 = max(start, pd.Timestamp(f'{year}-01-01')), min(pd.Timestamp(end), pd.Timestamp(f'{year}-12-31'))
        if y0 <= y1:
            images.append(download.search(start=y0, end=y1, area=box.bounds))
    images = pd.concat(images, ignore_index=True) if images else pd.DataFrame()
    if images.empty:
        return images
    t = times.time.values
    window = np.timedelta64(int(window_h * 3600), 's')
    rows = []
    for row in images.itertuples(index=False):
        tt = np.datetime64(row.time)
        sel = (t >= tt - window) & (t <= tt + window)
        if not sel.any():
            continue
        dt = (t[sel] - tt) / np.timedelta64(1, 'm')
        footprint = shape(row.footprint) if row.footprint else None
        if footprint is None or not footprint.contains(Point(info.lon, info.lat)):
            continue
        rows.append({**row._asdict(), 'n_obs': int(sel.sum()),
                     'n_obs_lev20': int((times.level.values[sel] == 2).sum()),
                     'dt_min': float(dt[np.argmin(np.abs(dt))]),
                     'coverage': footprint.intersection(box).area / box.area,
                     'same_zone': int(row.tile[:2]) == int((info.lon + 180) // 6) + 1})
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    # one tile per datatake: best coverage of the box (to 1 %), then an image on this machine (not downloaded),
    # then the UTM zone of the station
    plan['has_local'] = [local_safe(name, time) is not None for name, time in zip(plan.name, plan.time)]
    plan['cover'] = plan.coverage.round(2)
    plan = (plan.sort_values(['cover', 'has_local', 'same_zone'], ascending=False, kind='stable')
            .drop_duplicates(['satellite', 'time']).sort_values('time').reset_index(drop=True))
    plan['station'] = station
    # images of the same datatake already on this machine: read in place (any baseline), not downloaded
    local = [local_safe(name, time) for name, time in zip(plan.name, plan.time)]
    plan['cdse_name'] = plan['name']
    plan['local'] = [str(path) if path else '' for path in local]
    plan['name'] = [path.name if path else name for path, name in zip(local, plan.name)]
    plan['path'] = [str(subset_path(name, station, time)) for name, time in zip(plan.name, plan.time)]
    return plan.drop(columns=['footprint', 'cover', 'has_local'])


def read_plan(station):
    try:
        plan = pd.read_csv(PLANS / f'{station}.csv', parse_dates=['time'])
    except pd.errors.EmptyDataError:      # station without image
        return pd.DataFrame()
    if 'local' in plan:
        plan['local'] = plan['local'].fillna('')
    return plan


# ------------------------------------------------------------------------------------------------------------
# subset of one image
# ------------------------------------------------------------------------------------------------------------

def _packing(values, nbit=16):
    """int16 scale factor and offset covering the range of the values."""
    vmax, vmin = np.nanmax(values), np.nanmin(values)
    scale = (vmax - vmin) / (2 ** nbit - 2) or 1e-6
    return float(scale), float(vmin + 2 ** (nbit - 1) * scale)


def write_subset(prod, path, complevel=5):
    """GRSdriver product -> NetCDF, bands and angles packed in int16 (written to .part, then renamed)."""
    for obj in [prod] + [prod[v] for v in prod.variables]:
        obj.attrs = {k: (int(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in obj.attrs.items()
                     if v is not None}
    scale, offset = _packing(prod.bands.values)
    enc = dict(zlib=True, complevel=complevel)
    encoding = {'bands': {'dtype': 'int16', 'scale_factor': scale, 'add_offset': offset, '_FillValue': -32768, **enc},
                'vza': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, **enc},
                'raa': {'dtype': 'int16', 'scale_factor': 0.1, '_FillValue': -9999, **enc},
                'sza': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, **enc}}
    encoding = {v: encoding.get(v, dict(enc)) for v in prod.data_vars}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.part')
    prod.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


def write_quicklook(prod, path):
    """TOA RGB (665, 560, 490 nm) with the station (centre) marked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rgb = np.stack([prod.bands.sel(wl=wl, method='nearest').values for wl in (665, 560, 490)], axis=-1)
    rgb = np.clip((np.nan_to_num(rgb) / 0.25) ** (1 / 2.2), 0, 1)
    fig = plt.figure(figsize=(2.51, 2.51), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation='nearest')
    ny, nx = rgb.shape[:2]
    ax.plot(nx // 2, ny // 2, marker='+', color='red', markersize=10, mew=1.5)
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_item(prod, row, info, nc, png, valid_fraction):
    """STAC Item ``<name>.json`` next to the subset (assets relative to the Item), written last."""
    import GRSdriver
    from shapely.geometry import box, mapping
    from shapely.ops import transform
    from pyproj import Transformer

    crs = prod.rio.crs
    epsg = crs.to_epsg()
    bounds = prod.rio.bounds()
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    geometry = transform(to_lonlat, box(*bounds))
    tr = prod.rio.transform()
    attrs = prod.attrs
    match = _NAME.match(row['name'])
    properties = {
        'datetime': pd.Timestamp(row['time']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-4] + 'Z',
        'constellation': 'sentinel-2',
        'platform': str(attrs.get('DATATAKE_1_SPACECRAFT_NAME', match.group(1))).lower(),
        'instruments': ['msi'],
        'processing:level': 'L1C',
        'processing:software': {'GRSdriver': GRSdriver.__version__},
        'processing:datetime': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'grid:code': f"MGRS-{row['tile']}",
        'eo:cloud_cover': float(row['cloud_cover']) if pd.notna(row['cloud_cover']) else None,
        's2:processing_baseline': f'{match.group(3)[:2]}.{match.group(3)[2:]}',
        'sat:relative_orbit': int(match.group(4)),
        'proj:code': f'EPSG:{epsg}',
        'proj:shape': [int(prod.sizes['y']), int(prod.sizes['x'])],
        'proj:transform': [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
        'proj:bbox': list(bounds),
        'view:sun_elevation': float(90 - np.nanmean(prod.sza.values)),
        'view:incidence_angle': float(np.nanmean(prod.vza.values)),
        'subset:size_m': 2 * HALF,
        'subset:resolution_m': RESOLUTION,
        'subset:valid_fraction': round(float(valid_fraction), 4),
        'aeronet_oc:station': row['station'],
        'aeronet_oc:sites': list(info.sites),
        'aeronet_oc:lon': float(info.lon),
        'aeronet_oc:lat': float(info.lat),
        'aeronet_oc:n_obs': int(row['n_obs']),
        'aeronet_oc:n_obs_lev20': int(row['n_obs_lev20']),
        'aeronet_oc:dt_min': float(row['dt_min']),
        'aeronet_oc:window_h': WINDOW_H,
    }
    item = {
        'type': 'Feature', 'stac_version': '1.0.0', 'id': nc.stem,
        'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                            'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                            'https://stac-extensions.github.io/file/v2.1.0/schema.json',
                            'https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                            'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                            'https://stac-extensions.github.io/sat/v1.0.0/schema.json',
                            'https://stac-extensions.github.io/view/v1.0.0/schema.json'],
        'geometry': mapping(geometry), 'bbox': list(geometry.bounds),
        'properties': {k: v for k, v in properties.items() if v is not None},
        'links': [{'rel': 'derived_from', 'type': 'application/octet-stream',
                   'href': f"file://{row['local']}" if row.get('local') else f"s3:/{row['s3_path']}",
                   'title': (f"Sentinel-2 L1C image {row['name']}, "
                             + ('archive of this machine' if row.get('local') else 'CDSE eodata bucket'))}],
        'assets': {
            'data': {'href': f'./{nc.name}', 'type': 'application/x-netcdf', 'roles': ['data'],
                     'title': 'L1C subset (GRSdriver product: TOA reflectance, angles, SRF), GRS input',
                     'file:size': nc.stat().st_size},
            'thumbnail': {'href': f'./{png.name}', 'type': 'image/png', 'roles': ['thumbnail', 'overview'],
                          'title': 'TOA RGB, station at the centre', 'file:size': png.stat().st_size},
        },
    }
    path = nc.with_suffix('.json')
    tmp = path.with_name(f'.{path.name}.part')
    tmp.write_text(json.dumps(item, indent=2))
    os.replace(tmp, path)
    return path


def local_safe(name, time):
    """
    Image of the same datatake and tile on this machine (any processing baseline, the last processing if
    several), or None.
    """
    mission, sensing, _, orbit, tile, _ = _NAME.match(name).groups()
    folder = LOCAL_L1C / tile / f'{pd.Timestamp(time):%Y/%m/%d}'
    found = sorted(folder.glob(f'{mission}_MSIL1C_{sensing}_N*_R{orbit}_T{tile}_*.SAFE'),
                   key=lambda path: _NAME.match(path.name).group(6))
    found = [path for path in found if (path / 'manifest.safe').is_file()]
    return found[-1] if found else None


def _write_products(safe, row, info, driver_options=None):
    """GRSdriver subset of an image (local folder or URL) -> NetCDF, quicklook and STAC Item."""
    import GRSdriver

    nc = Path(row['path'])
    epsg = utm_epsg(row['tile'], info.lat)
    l1c = GRSdriver.Sentinel2Driver(str(safe), subset=[station_box(info.lon, info.lat, epsg)],
                                    resolution=RESOLUTION, resolution_angle=RESOLUTION_ANGLE, **(driver_options or {}))
    l1c.load_product()
    prod = l1c.prod.load()
    valid = np.isfinite(prod.bands.sel(wl=865).values) & (prod.bands.sel(wl=865).values > 0)
    if prod.sizes['x'] == 0 or prod.sizes['y'] == 0 or not valid.any():
        raise ValueError('no data in the subset (outside the swath)')
    write_subset(prod, nc)
    png = nc.with_name(nc.stem + '_ql.png')
    write_quicklook(prod, png)
    write_item(prod, row, info, nc, png, valid.mean())


def _download_and_write(row, info):
    """Download the image from CDSE, write its subset, delete the image (whatever happens)."""
    from grstbx import download

    result = download.download(pd.DataFrame([{**row, 'name': row['cdse_name']}]), str(WORK), workers=1)
    downloaded = Path(result.path.iloc[0])
    try:
        if result.status.iloc[0] == 'failed':
            raise OSError(f'download failed: {result.error.iloc[0]}')
        _write_products(downloaded, row, info)
    finally:                                   # only the downloaded images are deleted
        shutil.rmtree(downloaded, ignore_errors=True)
        shutil.rmtree(str(downloaded) + '.part', ignore_errors=True)
        for folder in list(downloaded.parents)[:4]:      # empty <tile>/YYYY/MM/DD folders
            try:
                folder.rmdir()
            except OSError:
                break


def make_subset(row, info, source='download'):
    """
    Subset of one image (one process): image of this machine, read in place, or else the image of CDSE,
    downloaded, subset and deleted (``source='download'``), or read in place from the CDSE S3 bucket
    (``source='remote'``: only the parts of the files in the box are read; downloaded if this fails, e.g. a
    JPEG2000 tile not decoded).
    """
    from grstbx import download

    nc = Path(row['path'])
    if nc.with_suffix('.json').exists():
        return row['name'], 'exists', 0.
    t0 = time.time()
    local = row.get('local') if isinstance(row.get('local'), str) else ''
    try:
        if local:
            _write_products(local, row, info)
            status = 'local'
        elif source == 'remote':
            url = 's3://eodata/' + row['s3_path'].lstrip('/').removeprefix('eodata/')
            try:
                _write_products(url, row, info, {'storage_options': download.cdse_storage_options()})
                status = 'remote'
            except Exception as exc:
                _log(f"{row['name']}: remote read failed ({type(exc).__name__}: {exc}), downloading it")
                _download_and_write(row, info)
                status = 'downloaded'
        else:
            _download_and_write(row, info)
            status = 'downloaded'
        Path(str(nc) + '.err').unlink(missing_ok=True)
    except Exception as exc:
        nc.parent.mkdir(parents=True, exist_ok=True)
        Path(str(nc) + '.err').write_text(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
        status = f'FAILED {type(exc).__name__}: {exc}'
    return row['name'], status, time.time() - t0


# ------------------------------------------------------------------------------------------------------------
# STAC catalogue
# ------------------------------------------------------------------------------------------------------------

def bucket_url(path):
    return f's3://{BUCKET}/{PREFIX}/{Path(path).relative_to(OUT).as_posix()}'


def build_catalog(out_dir=STAC_DIR):
    """
    Static STAC catalogue of the subsets (layout of grstbx.stac: catalog.json, <collection>/collection.json,
    items.json and <id>/<id>.json; relative links, assets as URLs in the bucket), read by
    ``grstbx.app.search_items``. Also writes ``stations.geojson`` at the root of the bucket.
    """
    out_dir = Path(out_dir)
    items = []
    for path in sorted(OUT.glob('*/*/*/*/*.json')):
        item = json.loads(path.read_text())
        for asset in item['assets'].values():
            asset['href'] = bucket_url(path.parent / asset['href'].removeprefix('./'))
        items.append(item)
    items.sort(key=lambda item: item['properties']['datetime'])
    media = 'application/json'
    for item in items:
        item['collection'] = COLLECTION
        item['links'] = [link for link in item['links'] if link['rel'] == 'derived_from'] + [
            {'rel': 'root', 'href': '../../catalog.json', 'type': media},
            {'rel': 'parent', 'href': '../collection.json', 'type': media},
            {'rel': 'collection', 'href': '../collection.json', 'type': media}]
        _write_json(out_dir / COLLECTION / item['id'] / f"{item['id']}.json", item)
    _write_json(out_dir / COLLECTION / 'items.json', {'type': 'FeatureCollection', 'features': items})
    times = [item['properties']['datetime'] for item in items]
    bbox = ([min(i['bbox'][0] for i in items), min(i['bbox'][1] for i in items),
             max(i['bbox'][2] for i in items), max(i['bbox'][3] for i in items)] if items else [-180, -90, 180, 90])
    station_list = sorted({item['properties']['aeronet_oc:station'] for item in items})
    _write_json(out_dir / COLLECTION / 'collection.json', {
        'type': 'Collection', 'stac_version': '1.0.0', 'id': COLLECTION,
        'title': 'Sentinel-2 L1C subsets around the AERONET-OC stations',
        'description': ('Sentinel-2 MSI L1C subsets of 5 km x 5 km (251 x 251 pixels at 20 m) centred on the '
                        'AERONET-OC stations, for the overpasses with AERONET-OC measurements within '
                        f'{WINDOW_H:g} h: GRSdriver products (TOA reflectance, sun and viewing angles per band, '
                        'SRF) in NetCDF, input of GRS for the matchups of the L2A Rrs with AERONET-OC.'),
        'license': 'proprietary',
        'extent': {'spatial': {'bbox': [bbox]},
                   'temporal': {'interval': [[min(times), max(times)] if times else [None, None]]}},
        'summaries': {'processing:level': ['L1C'], 'aeronet_oc:station': station_list,
                      'platform': sorted({item['properties']['platform'] for item in items})},
        'links': [{'rel': 'root', 'href': '../catalog.json', 'type': media},
                  {'rel': 'parent', 'href': '../catalog.json', 'type': media},
                  {'rel': 'items', 'href': './items.json', 'type': 'application/geo+json'}]
                 + [{'rel': 'item', 'href': f"./{item['id']}/{item['id']}.json", 'type': 'application/geo+json'}
                    for item in items]})
    _write_json(out_dir / 'catalog.json', {
        'type': 'Catalog', 'stac_version': '1.0.0', 'id': 'aeronet-oc',
        'title': 'Satellite subsets around the AERONET-OC stations',
        'description': 'Image subsets for the validation of GRS with AERONET-OC (Sentinel-2 L1C for now)',
        'links': [{'rel': 'root', 'href': './catalog.json', 'type': media},
                  {'rel': 'child', 'href': f'./{COLLECTION}/collection.json', 'type': media,
                   'title': 'Sentinel-2 L1C subsets'}]})
    # stations
    st = stations()
    counts = pd.Series([item['properties']['aeronet_oc:station'] for item in items]).value_counts()
    features = [{'type': 'Feature', 'id': name,
                 'geometry': {'type': 'Point', 'coordinates': [float(info.lon), float(info.lat)]},
                 'properties': {'station': name, 'sites': list(info.sites), 'n_aeronet_obs': int(info.n_obs),
                                'first': str(info['first']), 'last': str(info['last']),
                                'n_s2_subsets': int(counts.get(name, 0)),
                                'box': list(lonlat_box(info.lon, info.lat).bounds)}}
                for name, info in st.iterrows()]
    _write_json(DATA_ROOT / 'stations.geojson', {'type': 'FeatureCollection', 'features': features})
    return len(items)


def _write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.part')
    tmp.write_text(json.dumps(content, indent=1))
    os.replace(tmp, path)


def setup_bucket():
    """
    Folder of the bucket (served by grstbx-s3-bucket with the other buckets of /data/satellite/s3): links to
    the folders of DATA_ROOT that are published (sentinel-2/, stac/, stations.geojson), not to work/.
    """
    for folder in (OUT, L2A, STAC_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    BUCKET_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('sentinel-2', 'stac', 'stations.geojson'):
        link = BUCKET_DIR / name
        if not link.is_symlink():
            link.symlink_to(DATA_ROOT / name)


# ------------------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------------------

def _log(msg):
    print(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}', flush=True)


def plan_one(name, st, obs, a):
    """Plan of a station (kept if it exists, unless --overwrite-plan)."""
    path = PLANS / f'{name}.csv'
    if path.exists() and not a.overwrite_plan:
        _log(f'{name}: plan exists ({len(read_plan(name))} images)')
        return
    plan = plan_station(name, st.loc[name], obs, end=a.end)
    tmp = path.with_name(f'.{path.name}.part')
    plan.to_csv(tmp, index=False)
    os.replace(tmp, path)
    n_local = int((plan['local'].fillna('') != '').sum()) if len(plan) else 0
    _log(f'{name}: {len(plan)} images, {n_local} on this machine, '
         f'{plan.loc[plan["local"].fillna("") == "", "size"].sum() / 1e9 if len(plan) else 0:.0f} GB of L1C '
         'not on this machine')


def run_one(name, st, a):
    """Subsets of the images of the plan of a station, then the catalogue."""
    if not (PLANS / f'{name}.csv').exists():
        _log(f'{name}: no plan, skipped (run the step plan first)')
        return
    plan = read_plan(name)
    if plan.empty:
        _log(f'{name}: no image')
        return
    done = [Path(p).with_suffix('.json').exists() for p in plan.path]
    failed = [Path(p + '.err').exists() for p in plan.path]
    todo = plan[[not d and (a.retry_failed or not f) for d, f in zip(done, failed)]]
    if a.limit and a.source == 'remote':
        todo = todo[todo['local'] == ''].head(a.limit)
    elif a.limit:
        todo = todo.groupby(todo['local'] != '', group_keys=False).head(max(1, a.limit // 2))
    _log(f'{name}: {len(plan)} images, {sum(done)} done, {sum(failed)} failed, {len(todo)} to process')
    info = st.loc[name]
    n_ok = n_fail = 0
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(make_subset, row._asdict(), info, a.source) for row in todo.itertuples(index=False)]
        for i, future in enumerate(as_completed(futures), 1):
            image, status, dt = future.result()
            n_ok += not status.startswith('FAILED')
            n_fail += status.startswith('FAILED')
            _log(f'{name} [{i}/{len(todo)}] {status:10s} {image} ({dt:.0f} s)')
    _log(f'{name}: done, {n_ok} subsets written, {n_fail} failed')
    build_catalog()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('step', choices=['stations', 'plan', 'run', 'all', 'catalog'],
                    help='all: plan and run, station after station')
    ap.add_argument('--stations', nargs='*', help='stations (default: all, by decreasing number of measurements)')
    ap.add_argument('--end', default=None, help='last date of the plan (default: today)')
    ap.add_argument('--workers', type=int, default=3, help='images processed at the same time')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--overwrite-plan', action='store_true')
    ap.add_argument('--source', choices=['download', 'remote'], default='download',
                    help='images not on this machine: downloaded then deleted, or read in place in the CDSE bucket')
    ap.add_argument('--limit', type=int, default=None, help='at most this number of images per station (tests)')
    a = ap.parse_args(argv)

    obs = aeronet_obs()
    st = stations(obs)
    names = a.stations or list(st.index)
    if a.step == 'stations':
        pd.set_option('display.width', 200)
        print(st.to_string())
        return
    if a.step == 'catalog':
        setup_bucket()
        _log(f'{build_catalog()} items in the catalogue {STAC_DIR}')
        return
    PLANS.mkdir(parents=True, exist_ok=True)
    setup_bucket()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in names:
        if a.step in ('plan', 'all'):
            plan_one(name, st, obs, a)
        if a.step in ('run', 'all'):
            run_one(name, st, a)
    _log('finished')


if __name__ == '__main__':
    sys.exit(main())
