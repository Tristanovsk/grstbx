"""
Web dashboard to explore GRS (or hGRS) products: L2A images with :class:`~grstbx.visual.ViewSpectral`,
L2B images with :class:`~grstbx.visual.ViewParam`, one product, a time series of products or a
multi-temporal datacube.

The dashboard is a `Panel <https://panel.holoviz.org>`_ application served by a Python process, which
reads the pixels displayed on the screen and the spectra or time series under the mouse: start it with

.. code-block:: console

    grstbx-viewer                                   # empty dashboard, choose the products in the sidebar
    grstbx-viewer /data/S2A_MSIL2AGRS_..._V3.0.1.zarr --root /data/satellite
    grstbx-viewer "/data/L2B/BEFR/2024/*/*/*.nc"    # time series of L2B products (glob pattern)
    grstbx-viewer /data/L2B --tile 31TFJ            # L2B folders <tile>/YYYY/MM/DD/<product>/<product>.nc
    grstbx-viewer "/data/L2B/31TFJ/2021/*/*/*/*.json" --max-cloud-cover 20   # STAC Items of the products
    grstbx-viewer --stac /data/L2B                  # STAC search card on the Items of a folder (or a STAC API)
    grstbx-viewer /data/L2A_zarr --tile 31TEJ       # time series of a tile, folders <tile>/YYYY/MM/DD/<product>
    grstbx-viewer --address 0.0.0.0 --port 5006 --allow-websocket-origin myserver:5006

or from Python with :func:`serve`, or inline in a Jupyter notebook with :func:`panel`. The products are
read on the machine running the server: with ``--address 0.0.0.0``, the dashboard is reachable from other
computers of the network.
"""

import argparse
import glob
import io
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from .driver import is_hgrs, is_zarr

logger = logging.getLogger(__name__)

ACCEPTED = ('GRS L2A or L2B Zarr stores (`*.zarr`), GRS L2A NetCDF product folders (`<name>/` with '
            '`<name>.nc` and `<name>_anc.nc`), GRS L2B NetCDF files or product folders (`<name>/` with '
            '`<name>.nc`, `<name>.log` and `<name>.json`) or their STAC Items (`*.json`), hGRS L2A NetCDF files (EnMAP, PRISMA), '
            'folders of products (e.g. `<tile>/YYYY/MM/DD/<product>`), or a NetCDF datacube with a `time` '
            'dimension')

# variables of the L2B products that are not parameters to display
NOT_PARAMETERS = ('flags', 'mask', 'spatial_ref', 'dem', 'valid_pix_prop')
# order of the L2B parameters in the viewer (by prefix), the other ones come last
PARAMETER_ORDER = ('Chla', 'SPM', 'TURB', 'acdom', 'Kd')
# datacubes smaller than this (bytes) are loaded in memory: fast time series under the mouse and exports
MAX_IN_MEMORY = 1e9
# above this number of dates, the time series under the mouse (one block read per date) is off at opening
LIVE_MAX_DATES = 30


# -- opening of the products ------------------------------------------------------------------------

# product names: <satellite>_<level>_<acquisition time>_..._T<tile>_<processing time>[_V<version>]
_DATETIME = re.compile(r'(\d{8}T\d{6})')
_TILE = re.compile(r'_T(\d{2}[A-Z]{3})(?=_|$)')
_VERSION = re.compile(r'_[Vv](\d[\w.]*)$')


# columns of the product tables (see product_info)
COLUMNS = ['path', 'name', 'satellite', 'level', 'tile', 'time', 'processing_time', 'version', 'cloud_cover',
           'crs', 'bbox', 'item']


def _read_item(path):
    """STAC Item of a JSON file (dict), or None if ``path`` is not a STAC Item."""
    path = str(path)
    if not (path.endswith('.json') and os.path.isfile(path)):
        return None
    try:
        with open(path) as f:
            item = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(item, dict) or item.get('type') != 'Feature' or 'stac_version' not in item:
        return None
    return item


def _data_href(item, item_path=None):
    """Product of a STAC Item: href of its ``data`` asset (or first asset with the role ``data``), relative
    hrefs resolved from the folder of the Item (local file ``item_path``) or from its ``self`` link; None if
    there is none."""
    assets = item.get('assets', {})
    asset = assets.get('data') or next((a for a in assets.values() if 'data' in a.get('roles', [])), None)
    if not asset or 'href' not in asset:
        return None
    href = asset['href']
    if '://' in href or os.path.isabs(href):
        return href
    if item_path is None:
        base = next((link['href'] for link in item.get('links', []) if link.get('rel') == 'self'), None)
        return urllib.parse.urljoin(base, href) if base else href
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(item_path)), href))


def _item_of(path):
    """
    Path of the STAC Item describing a product: the path itself (Item JSON file), ``<name>/<name>.json`` in
    a product folder (GRSl2bgen ``--product_dir --stac``) or ``<name>.json`` next to ``<name>.nc`` /
    ``<name>.zarr``; None if there is none.
    """
    path = str(path).rstrip('/')
    if path.endswith('.json'):
        return path if _read_item(path) else None
    if os.path.isdir(path) and not is_zarr(path):
        candidate = os.path.join(path, os.path.basename(path) + '.json')
    else:
        candidate = re.sub(r'\.(zarr|nc)$', '', path) + '.json'
    return candidate if _read_item(candidate) else None


def is_stac_item(path):
    """True if ``path`` is a STAC Item JSON file whose data asset exists (or is a URL)."""
    item = _read_item(path)
    href = _data_href(item, path) if item else None
    return bool(href) and ('://' in href or os.path.exists(href))


def item_info(path):
    """
    Description of a product from its STAC Item (see :func:`product_info`): ``path`` is the data asset,
    ``time`` the Item ``datetime``, ``tile`` its ``grid:code`` (``MGRS-<tile>``), ``level`` its
    ``processing:level``, ``version`` the GRS version of ``processing:software``, ``cloud_cover`` its
    ``eo:cloud_cover`` (%), ``crs`` its ``proj:code``; the missing fields are taken from the name of the
    product.

    :param path: Item JSON file, or Item (dict, e.g. from a STAC API: see :func:`search_items`)
    :return: dict, see :func:`product_info`
    """
    if isinstance(path, dict):
        item, path = path, None
    else:
        item = _read_item(path)
        if item is None:
            raise ValueError(f'{path} is not a STAC Item')
    href = _data_href(item, path)
    info = _name_info(item.get('id') or href or path)
    props = item.get('properties', {})
    platform = str(props.get('platform', ''))
    tile = str(props.get('grid:code', ''))
    software = props.get('processing:software') or {}
    if props.get('datetime'):
        info['time'] = pd.Timestamp(props['datetime']).tz_localize(None)
    info.update({k: v for k, v in {
        'path': _product_file(href) if href and '://' not in href else href,
        'satellite': ('S2' + platform[-1].upper()) if re.fullmatch(r'sentinel-2[a-d]', platform.lower()) else None,
        'level': props.get('processing:level'),
        'tile': tile[len('MGRS-'):] if tile.startswith('MGRS-') else None,
        'version': str(software['GRS']) if 'GRS' in software else None,
        'cloud_cover': props.get('eo:cloud_cover'),
        'crs': props.get('proj:code') or (f"EPSG:{props['proj:epsg']}" if props.get('proj:epsg') else None),
        'bbox': list(item['bbox']) if item.get('bbox') else None,
    }.items() if v is not None})
    info['item'] = os.path.abspath(path) if path else next(
        (link['href'] for link in item.get('links', []) if link.get('rel') == 'self'), item.get('id'))
    return info


def _product_file(path):
    """
    File or store to open for a product folder ``<name>/``: its Zarr store ``<name>.zarr`` (GRS >= 3, or
    GRSl2bgen with ``--product_dir``), or its NetCDF file ``<name>.nc`` when there is no ancillary file
    ``<name>_anc.nc`` (GRSl2bgen L2B product folder, with ``<name>.log`` and the STAC Item ``<name>.json``).
    A GRS 2 L2A folder (``<name>.nc`` and ``<name>_anc.nc``) and any other path are returned unchanged.
    """
    path = str(path).rstrip('/')
    if is_zarr(path) or not os.path.isdir(path):
        return path
    base = os.path.join(path, os.path.basename(path))
    if os.path.isdir(base + '.zarr'):
        return base + '.zarr'
    if os.path.isfile(base + '.nc') and not os.path.isfile(base + '_anc.nc'):
        return base + '.nc'
    return path


def is_product(path):
    """
    True if ``path`` is a product opened by :func:`open_products`: Zarr store, NetCDF file, or product
    folder ``<name>/`` holding ``<name>.zarr`` (GRS >= 3) or ``<name>.nc`` (GRS 2 L2A, with
    ``<name>_anc.nc``, or GRSl2bgen L2B, see :func:`_product_file`), or STAC Item of a product (see
    :func:`is_stac_item`).
    """
    path = str(path).rstrip('/')
    name = os.path.basename(path)
    if path.endswith('.json'):
        return is_stac_item(path)
    return is_zarr(path) or (os.path.isfile(path) and path.endswith('.nc')) or (
        os.path.isdir(path) and (os.path.isfile(os.path.join(path, name + '.nc'))
                                 or os.path.isdir(os.path.join(path, name + '.zarr'))))


def _name_info(path):
    """Description of a product from its name only (see :func:`product_info`)."""
    stem = re.sub(r'\.(zarr|nc|json)$', '', os.path.basename(str(path).rstrip('/')))
    times = _DATETIME.findall(stem)
    tile, version = _TILE.search(stem), _VERSION.search(stem)
    return {'path': _product_file(path), 'name': stem, 'satellite': stem.split('_')[0],
            'level': 'L2B' if 'L2B' in stem.upper() else 'L2A',
            'tile': tile.group(1) if tile else None,
            'time': pd.Timestamp(times[0]) if times else pd.NaT,
            'processing_time': pd.Timestamp(times[1]) if len(times) > 1 else pd.NaT,
            'version': version.group(1) if version else '',
            'cloud_cover': np.nan, 'crs': None, 'bbox': None, 'item': None}


def product_info(path):
    """
    Description of a product: from its STAC Item when it has one (see :func:`item_info` and
    :func:`_item_of`), otherwise from its name (ESA convention
    ``<satellite>_<level>_<acquisition time>_..._T<tile>_<processing time>[_V<version>]``).

    :return: dict with ``path`` (file or store to open), ``name``, ``satellite``, ``level`` (``'L2A'`` or
             ``'L2B'``), ``tile``, ``time`` (acquisition), ``processing_time``, ``version``,
             ``cloud_cover`` (%, STAC Items only), ``crs`` (STAC Items only), ``item`` (path of the STAC
             Item; missing fields are None, NaN or NaT)
    """
    item = _item_of(path)
    return item_info(item) if item else _name_info(path)


def find_products(directory, maxdepth=6):
    """
    Products stored under ``directory``, e.g. in the tree ``<tile>/YYYY/MM/DD/<product>`` of the GRS batches
    (``directory`` can be the root, a tile, a year or a month folder). The folders of the products are not
    explored. The products with a STAC Item are described from it (see :func:`product_info`).

    :param directory: folder to explore
    :param maxdepth: maximum depth of the products below ``directory``
    :return: pandas.DataFrame of :func:`product_info`, one row per product, sorted by acquisition time
    """
    found = []

    def explore(folder, depth):
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return
        for name in names:
            path = os.path.join(folder, name)
            if is_product(path):
                found.append(product_info(path))
            elif os.path.isdir(path) and not name.startswith('.') and depth < maxdepth:
                explore(path, depth + 1)

    explore(str(directory), 0)
    # a product and its STAC Item side by side (flat layout) are found twice
    return (pd.DataFrame(found, columns=COLUMNS).drop_duplicates('path')
            .sort_values('time', kind='stable').reset_index(drop=True))


def select_products(products, tile=None, start=None, end=None, last_processing=True, max_cloud_cover=None):
    """
    Selection of products listed by :func:`find_products`.

    :param products: pandas.DataFrame of :func:`product_info`
    :param tile: tile to keep (e.g. ``'31TEJ'``), None: all
    :param start: first acquisition date (included), None: no limit
    :param end: last acquisition date (included), None: no limit
    :param last_processing: keep only the last processing of each acquisition (same satellite, tile and
                            acquisition time): latest processing time, release preferred to a ``dev`` version
    :param max_cloud_cover: maximum cloud cover (%, ``eo:cloud_cover`` of the STAC Items); the products
                            without cloud cover (no STAC Item) are kept. None: no limit
    :return: pandas.DataFrame, sorted by acquisition time
    """
    selected = products
    if max_cloud_cover is not None and 'cloud_cover' in selected:
        cloud = pd.to_numeric(selected.cloud_cover, errors='coerce')
        selected = selected[cloud.isna() | (cloud <= max_cloud_cover)]
    if tile:
        selected = selected[selected.tile == tile]
    if start is not None:
        selected = selected[selected.time >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected.time < pd.Timestamp(end) + pd.Timedelta('1D')]
    if last_processing and len(selected):
        release = ~selected.version.str.contains('dev', case=False)
        selected = (selected.assign(_release=release)
                    .sort_values(['time', 'processing_time', '_release'], kind='stable')
                    .drop_duplicates(['satellite', 'tile', 'time', 'level'], keep='last')
                    .drop(columns='_release'))
    return selected.sort_values('time', kind='stable').reset_index(drop=True)


def _read_json(url, timeout=60):
    """JSON document of a URL (``http(s)://``, or ``s3://`` and the like through fsspec) or local file."""
    if re.match(r'https?://', url):
        req = urllib.request.Request(url, headers={'Accept': 'application/geo+json, application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    import fsspec

    with fsspec.open(url, 'r') as f:
        return json.load(f)


def _fetch_item(url, timeout=60):
    """STAC Item at a URL (dict), with a ``self`` link to the URL so that its relative hrefs resolve."""
    item = _read_json(url, timeout)
    if not isinstance(item, dict) or item.get('type') != 'Feature':
        raise ValueError(f'{url} is not a STAC Item')
    if not any(link.get('rel') == 'self' for link in item.get('links', [])):
        item.setdefault('links', []).append({'rel': 'self', 'href': url})
    return item


def _is_static_catalog(source):
    """True if ``source`` is the ``catalog.json`` or ``collection.json`` of a static STAC catalogue."""
    if not (isinstance(source, str) and source.strip().endswith(('catalog.json', 'collection.json'))):
        return False
    return '://' in source or os.path.isfile(source.strip())


def _join(base, href):
    """Absolute location of ``href`` relative to the document ``base`` (URL of any scheme or local path)."""
    if '://' in href or os.path.isabs(href):
        return href
    if '://' not in base:
        return os.path.normpath(os.path.join(os.path.dirname(base), href))
    scheme, rest = base.split('://', 1)
    return f'{scheme}://' + os.path.normpath(os.path.join(os.path.dirname(rest), href))


def _static_items(source, collections=None):
    """
    Items of a static STAC catalogue, following its ``child`` links: the ``items`` link of a collection,
    read at once (see :mod:`grstbx.stac`), or else its ``item`` links, one request per Item.

    :param source: ``catalog.json`` or ``collection.json`` (URL or local file)
    :param collections: collection id or list of ids to keep (default: all)
    :return: list of Items (dict), with a ``self`` link to their location
    """
    collections = [collections] if isinstance(collections, str) else collections
    items = []
    node = _read_json(source)
    links = [(link, source) for link in node.get('links', [])]
    wanted = node.get('type') != 'Collection' or not collections or node.get('id') in collections
    while links:
        link, base = links.pop(0)
        href = _join(base, link['href'])
        if link['rel'] == 'child':
            child = _read_json(href)
            if child.get('type') != 'Collection' or not collections or child.get('id') in collections:
                links += [(sub, href) for sub in child.get('links', [])]
        elif link['rel'] == 'items' and wanted:
            features = _read_json(href).get('features', [])
            links = [(sub, sub_base) for sub, sub_base in links
                     if not (sub['rel'] == 'item' and sub_base == base)]
            for item in features:
                location = _join(href, f"{item['id']}/{item['id']}.json")
                item['links'] = [*item.get('links', []), {'rel': 'self', 'href': location}]
            items += features
        elif link['rel'] == 'item' and wanted:
            items.append(_fetch_item(href))
    return items


def _stac_api_items(url, body, limit, timeout=60):
    """Items of a STAC API ``POST <url>/search`` (Item Search), following the ``next`` links."""
    url = url.rstrip('/')
    request = {'url': url if url.endswith('/search') else url + '/search', 'method': 'POST', 'body': body}
    items = []
    while request and len(items) < limit:
        data = json.dumps(request['body']).encode() if request.get('body') is not None else None
        method = request.get('method', 'POST' if data else 'GET')
        req = urllib.request.Request(request['url'], data=data if method == 'POST' else None, method=method,
                                     headers={'Content-Type': 'application/json',
                                              'Accept': 'application/geo+json, application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            page = json.load(response)
        items += page.get('features', [])
        link = next((link for link in page.get('links', []) if link.get('rel') == 'next'), None)
        if not link or not page.get('features'):
            break
        # POST paging: the body of the link replaces (merge=False) or updates (merge=True) the search
        body = link.get('body')
        if body is not None and link.get('merge'):
            body = {**request['body'], **body}
        request = {'url': link['href'], 'method': link.get('method', 'GET'), 'body': body}
    return items[:limit]


def search_items(source, bbox=None, intersects=None, start=None, end=None, max_cloud_cover=None, tile=None,
                 collections=None, version=None, limit=1000):
    """
    Products of a STAC catalogue matching an area, a period, a tile and a maximum cloud cover.

    The catalogue is either a STAC API (URL ``http(s)://...``: Item Search ``POST /search``, with the
    area, period, collections and cloud cover sent to the server, ``eo:cloud_cover`` through the query
    extension), a static catalogue (its ``catalog.json`` or ``collection.json``, in a bucket or on disk, see
    :mod:`grstbx.stac`), or local STAC Items: folders, Item files or glob patterns (see :func:`catalogue`),
    e.g. the products of GRSl2bgen ``--product_dir --stac``. The criteria are applied again to the results,
    so a server ignoring the query extension gives the same selection.

    :param source: STAC API URL, ``catalog.json`` of a static catalogue (e.g.
                   ``s3://bucket/stac/catalog.json``), or folder(s), Item file(s) or glob pattern(s)
    :param bbox: area (lon_min, lat_min, lon_max, lat_max), deg
    :param intersects: area as a shapely geometry or GeoJSON dict (lon/lat), instead of ``bbox``
    :param start: first acquisition date (included)
    :param end: last acquisition date (included)
    :param max_cloud_cover: maximum cloud cover (%); the Items without ``eo:cloud_cover`` are kept
    :param tile: tile (e.g. ``'31TFJ'``, Items with ``grid:code``)
    :param collections: STAC API or static catalogue: collection id or list of ids
    :param version: GRS version(s) to keep (``version`` of :func:`product_info`, from ``processing:software``
                    or the name, e.g. ``'3.1.0'``; a leading ``V`` is ignored), str or list
    :param limit: maximum number of Items read from a STAC API
    :return: pandas.DataFrame of :func:`product_info` (``path``: data asset), sorted by acquisition time;
             only the products with a STAC Item
    """
    from shapely.geometry import box, mapping, shape

    if intersects is not None and isinstance(intersects, dict):
        intersects = shape(intersects)
    area = intersects if intersects is not None else (box(*bbox) if bbox is not None else None)
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) + pd.Timedelta('1D') - pd.Timedelta('1ms') if end is not None else None

    if _is_static_catalog(source):
        products = pd.DataFrame([item_info(item) for item in _static_items(source.strip(), collections)],
                                columns=COLUMNS)
    elif isinstance(source, str) and re.match(r'https?://', source.strip()):
        body = {'limit': min(limit, 100)}
        if intersects is not None:
            body['intersects'] = mapping(intersects)
        elif bbox is not None:
            body['bbox'] = list(bbox)
        if start is not None or end is not None:
            body['datetime'] = (f'{start.isoformat() + "Z" if start is not None else ".."}/'
                                f'{end.isoformat() + "Z" if end is not None else ".."}')
        if collections:
            body['collections'] = [collections] if isinstance(collections, str) else list(collections)
        if max_cloud_cover is not None:
            body['query'] = {'eo:cloud_cover': {'lte': max_cloud_cover}}
        rows = [item_info(item) for item in _stac_api_items(source.strip(), body, limit)]
        products = pd.DataFrame(rows, columns=COLUMNS)
    else:
        products = catalogue(source)
        products = products[products.item.notna()]

    if len(products):
        if area is not None:
            inside = products.bbox.map(lambda b: b is not None and box(*b).intersects(area))
            products = products[inside]
        if start is not None:
            products = products[products.time >= start.tz_localize(None)]
        if end is not None:
            products = products[products.time <= end.tz_localize(None)]
        if tile:
            products = products[products.tile == tile]
        if max_cloud_cover is not None:
            cloud = pd.to_numeric(products.cloud_cover, errors='coerce')
            products = products[cloud.isna() | (cloud <= max_cloud_cover)]
        if version:
            versions = [version] if isinstance(version, str) else list(version)
            products = products[products.version.isin([v.lstrip('Vv') for v in versions])]
    return products.sort_values('time', kind='stable').reset_index(drop=True)


def _utm_zone(tile):
    """UTM zone and hemisphere of a Sentinel-2 tile (e.g. '31TEJ' -> '31N')."""
    return tile[:2] + ('N' if tile[2] >= 'N' else 'S')


class SeveralZones(ValueError):
    """The selected products are in several UTM zones: a tile must be chosen (see ``tiles``)."""

    def __init__(self, tiles):
        self.tiles = tiles
        listing = ', '.join(f'{tile} ({count} product{"s" if count > 1 else ""})' for tile, count in tiles.items())
        super().__init__(f'the products are in several UTM zones: choose a tile among {listing}')


def expand_paths(paths):
    """
    Products and folders from paths and glob patterns.

    :param paths: path, glob pattern (e.g. ``'/data/L2B/BEFR/2024/*/*/*.nc'``), or list of them, or text
                  with one per line
    :return: list of existing paths (patterns expanded and sorted), without duplicates
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = str(paths).splitlines()
    out = []
    for item in paths:
        item = str(item).strip().rstrip('/')
        if not item:
            continue
        if '://' in item:
            # remote product (e.g. data asset of a STAC API Item), opened by xarray / fsspec
            if item not in out:
                out.append(item)
            continue
        matches = sorted(glob.glob(item)) if glob.has_magic(item) else [item]
        if not matches:
            raise FileNotFoundError(f'no product matches {item}')
        for match in matches:
            if not os.path.exists(match):
                raise FileNotFoundError(match)
            if match not in out:
                out.append(match)
    return out


def catalogue(paths):
    """
    Products of paths, glob patterns and folders (explored with :func:`find_products`).

    :param paths: see :func:`expand_paths`; URLs (``http(s)://``, ``s3://``) are remote Zarr stores or NetCDF
                  files (by their extension), or remote STAC Items (fetched and described from the Item)
    :return: pandas.DataFrame of :func:`product_info`, sorted by acquisition time
    """
    tables = []
    for path in expand_paths(paths):
        if '://' in path and not re.search(r'\.(zarr|nc)$', path):
            # remote STAC Item (e.g. result of a STAC API search, ``.../items/<id>``): described from it
            tables.append(pd.DataFrame([item_info(_fetch_item(path))]))
        elif '://' in path:
            # remote product (Zarr store or NetCDF file)
            tables.append(pd.DataFrame([_name_info(path)]))
        elif is_product(path):
            tables.append(pd.DataFrame([product_info(path)]))
        elif os.path.isdir(path):
            tables.append(find_products(path))
        elif path.endswith('.json'):
            # JSON file other than a STAC Item, e.g. zarr.json of a Zarr store matched by a glob pattern
            continue
        else:
            raise ValueError(f'{path} is not a product nor a folder of products')
    products = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=COLUMNS)
    return products.drop_duplicates('path').sort_values('time', kind='stable').reset_index(drop=True)


def _image_time(ds, path):
    """Acquisition time of an image: ``time`` coordinate, attributes or date in the product name."""
    if 'time' in ds.coords and ds['time'].size == 1:
        return np.datetime64(pd.Timestamp(ds['time'].values.ravel()[0]).tz_localize(None), 'ns')
    for attr in ('acquisition_date', 'start_date', 'PRODUCT_START_TIME'):
        if attr in ds.attrs:
            return np.datetime64(pd.Timestamp(ds.attrs[attr]).tz_localize(None), 'ns')
    match = _DATETIME.search(os.path.basename(path))
    if match:
        return np.datetime64(pd.Timestamp(match.group(1)), 'ns')
    raise ValueError(f'no acquisition time found in {path}')


def _open_level(path, level=0, root=None):
    """One resolution level of a product, as a lazy xarray.Dataset (``root``: root group of a Zarr store,
    already open)."""
    from .driver import open_zarr_image, open_hgrs_l2a
    from .subset import open_l2a

    if is_zarr(path):
        return open_zarr_image(path, level=level, root=root)
    if '://' in path:
        # remote NetCDF file: read through fsspec, the netCDF4 library has its own access to the buckets
        import fsspec

        return xr.open_dataset(fsspec.open(path).open(), engine='h5netcdf', decode_coords='all', chunks={})
    if is_hgrs(path):
        return open_hgrs_l2a(path)[0]
    if os.path.isdir(path):
        return open_l2a(path)[0]
    return xr.open_dataset(path, decode_coords='all', chunks={})


def _is_l2a(ds):
    return 'Rrs' in ds and 'wl' in ds['Rrs'].dims


def _parameters(ds, dims=('y', 'x')):
    """Variables of an L2B product (or datacube) displayed by ViewParam: float maps."""
    names = [name for name, var in ds.data_vars.items()
             if set(var.dims) == set(dims) and var.dtype.kind == 'f' and name not in NOT_PARAMETERS]

    def rank(name):
        return next((i for i, prefix in enumerate(PARAMETER_ORDER) if name.startswith(prefix)), len(PARAMETER_ORDER))

    return sorted(names, key=rank)


def _layers(ds, kind, params=None, mask_invalid=True):
    """
    Layers displayed by the viewers: Rrs (L2A) or the parameters (L2B); with ``mask_invalid``, the pixels
    with ``mask != 0`` (not valid water pixels) are set to NaN. The parameters missing in a product are
    filled with NaN.
    """
    if kind == 'L2A':
        rrs = ds['Rrs']
        if mask_invalid and 'mask' in ds:
            rrs = rrs.where(ds['mask'] == 0)
        return rrs
    present = [param for param in params if param in ds]
    if not present:
        raise ValueError('none of the parameters is in the product')
    layers = ds[present]
    if mask_invalid and 'mask' in ds:
        layers = layers.where(ds['mask'] == 0)
    for param in params:
        if param not in layers:
            layers[param] = xr.full_like(layers[present[0]], np.nan)
    return layers[params]


class _LazyLevels:
    """
    Resolution levels of a product, each opened on first access (sequence used by
    :class:`~grstbx.visual.Multiscale`): a long time series is opened instantly, the products being read
    only when they are displayed.
    """

    def __init__(self, path, kind, params=None, mask_invalid=True):
        self.path, self.kind, self.params, self.mask_invalid = path, kind, params, mask_invalid
        self._levels = {}
        self._n = None
        self._root = None

    @property
    def root(self):
        """Root group of a Zarr store, read once for all its levels (None for the other products)."""
        if self._root is None and is_zarr(self.path):
            import zarr

            self._root = zarr.open_group(self.path, mode='r')
        return self._root

    def __len__(self):
        if self._n is None:
            from .driver import _levels

            self._n = max(len(_levels(self.root)), 1) if is_zarr(self.path) else 1
        return self._n

    def resolution(self, level):
        """
        Pixel size of a level from the ``multiscales`` attribute of the store (GRS: scale from the full
        resolution; GRSl2bgen: scale from the previous level), without opening the level; None when the
        attribute does not give it.
        """
        from .driver import multiscales_layout

        layout = multiscales_layout(self.root.attrs) if self.root is not None else []
        if level == 0 or len(layout) <= level:
            return None
        res = abs(float(self[0].x[1] - self[0].x[0]))
        if isinstance(layout[level].get('scale'), (int, float)):
            # GRS: scale of the level from the full resolution
            return res * float(layout[level]['scale'])
        for item in layout[1:level + 1]:
            scale = (item.get('transform') or {}).get('scale')
            if not scale or len(set(scale)) != 1:
                return None
            res *= float(scale[0])
        return res

    def __getitem__(self, level):
        if level < 0:
            level += len(self)
        if not 0 <= level < len(self):
            raise IndexError(level)
        if level not in self._levels:
            self._levels[level] = _layers(_open_level(self.path, level, self.root), self.kind, self.params,
                                          self.mask_invalid)
        return self._levels[level]


# -- opening of an area over several tiles -------------------------------------------------------------

#: area of interest of a pin (smaller areas are enlarged to this size, m)
AREA_MIN_EXTENT = 5000.
#: largest side of the grid of an area (full-resolution pixels) for products without pyramid: their coarser
#: levels are computed from the whole area at full resolution
AREA_MAX_PIXELS = 4096
#: largest side of the grid of an area (full-resolution pixels) for products with pyramids: each level of the
#: area is made of the same level of the products, and only the window displayed is read
AREA_MAX_PIXELS_PYRAMID = 65536
#: side of the blocks (pixels) reprojected from the products of another UTM zone, each when it is displayed
AREA_BLOCK = 512


def _crs_of(row):
    """CRS of a product from its table row: STAC ``proj:code``, or UTM zone of its tile."""
    import pyproj

    if isinstance(row.crs, str) and row.crs:
        return pyproj.CRS(row.crs)
    if isinstance(row.tile, str) and re.fullmatch(r'\d{2}[A-Z]{3}', row.tile):
        zone = _utm_zone(row.tile)
        return pyproj.CRS.from_epsg((32600 if zone[-1] == 'N' else 32700) + int(zone[:2]))
    return None


def _area_bounds(area, crs, min_extent=AREA_MIN_EXTENT):
    """Bounds of an area (shapely geometry, lon/lat) in ``crs``, enlarged to at least ``min_extent`` (m)."""
    from rasterio.warp import transform_bounds

    xmin, ymin, xmax, ymax = transform_bounds('EPSG:4326', crs, *area.bounds, densify_pts=21)
    if xmax - xmin < min_extent:
        xmin, xmax = (xmin + xmax - min_extent) / 2, (xmin + xmax + min_extent) / 2
    if ymax - ymin < min_extent:
        ymin, ymax = (ymin + ymax - min_extent) / 2, (ymin + ymax + min_extent) / 2
    return xmin, ymin, xmax, ymax


def _area_grid(bounds, crs, res, template=None):
    """
    Grid of an area: pixel centres (x ascending, y descending) in ``crs`` with the pixel size ``res``, aligned
    on the pixels of ``template`` (a level of a product in ``crs``: the tiles of a UTM zone share its grid).

    :return: dict with crs, x, y, transform
    """
    from rasterio.transform import Affine

    xoff = float(template.x[0]) % res if template is not None else res / 2
    yoff = float(template.y[0]) % res if template is not None else res / 2
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(np.floor((xmin - xoff) / res) * res + xoff, xmax + res / 2, res)
    ys = np.arange(np.ceil((ymax - yoff) / res) * res + yoff, ymin - res / 2, -res)
    transform = Affine(res, 0, xs[0] - res / 2, 0, -res, ys[0] + res / 2)
    return {'crs': crs, 'x': xs, 'y': ys, 'transform': transform}


def _area_grids(area, crs, template):
    """
    Grids of the levels of an area, one per level of the pyramid of ``template`` (:class:`_LazyLevels` of a
    product), with the pixel size of its levels; aligned on its pixels when it is in ``crs``.
    """
    import pyproj

    aligned = pyproj.CRS(template[0].rio.crs) == crs
    nlevels = len(template)
    bounds = _area_bounds(area, crs)
    res = abs(float(template[0].x[1] - template[0].x[0]))
    grids = []
    for level in range(nlevels):
        level_res = (template.resolution(level) or abs(float(template[level].x[1] - template[level].x[0]))
                     if level else res)
        grids.append(_area_grid(bounds, crs, level_res, template[level] if aligned else None))
    largest = AREA_MAX_PIXELS_PYRAMID if nlevels > 1 else AREA_MAX_PIXELS
    nx, ny = len(grids[0]['x']), len(grids[0]['y'])
    if max(nx, ny) > largest:
        raise ValueError(f'the area is too large to be opened alone ({nx} x {ny} pixels, at most {largest}'
                         f'{"" if nlevels > 1 else " for products without pyramid"}): draw a smaller area, or open '
                         'the whole products')
    return grids


def _warp(da, grid, block=AREA_BLOCK):
    """
    Lazy reprojection (nearest neighbour) of a layer (..., y, x) onto the grid of an area: each block of
    ``block`` x ``block`` pixels of the grid is reprojected from its window of the layer when it is read.
    """
    import dask.array as dsa
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject, transform_bounds

    da = da.transpose(..., 'y', 'x')
    xs, ys, crs = grid['x'], grid['y'], grid['crs']
    res = xs[1] - xs[0]
    src_crs, sx, sy = da.rio.crs, da.x.values, da.y.values
    sdx, sdy = sx[1] - sx[0], sy[1] - sy[0]
    lead = da.shape[:-2]
    dtype = np.result_type(da.dtype, np.float32)

    def warp_block(_, block_info=None):
        (y0, y1), (x0, x1) = block_info[None]['array-location'][-2:]
        out = np.full(lead + (y1 - y0, x1 - x0), np.nan, dtype)
        bounds = transform_bounds(crs, src_crs, xs[x0] - res / 2, ys[y1 - 1] - res / 2, xs[x1 - 1] + res / 2,
                                  ys[y0] + res / 2, densify_pts=21)
        # window of the layer covering the block, with a margin of 2 pixels
        ix = np.nonzero((sx >= bounds[0] - 2 * abs(sdx)) & (sx <= bounds[2] + 2 * abs(sdx)))[0]
        iy = np.nonzero((sy >= bounds[1] - 2 * abs(sdy)) & (sy <= bounds[3] + 2 * abs(sdy)))[0]
        if len(ix) < 2 or len(iy) < 2:
            return out
        window = da.isel(x=slice(ix[0], ix[-1] + 1), y=slice(iy[0], iy[-1] + 1))
        values = np.asarray(window.compute(scheduler='synchronous').values, dtype)
        reproject(values, out, src_transform=Affine(sdx, 0, sx[ix[0]] - sdx / 2, 0, sdy, sy[iy[0]] - sdy / 2),
                  src_crs=src_crs, dst_transform=Affine(res, 0, xs[x0] - res / 2, 0, -res, ys[y0] + res / 2),
                  dst_crs=crs, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        return out

    blocks = dsa.empty(lead + (len(ys), len(xs)), chunks=lead + (block, block), dtype=dtype)
    coords = {dim: da[dim].values for dim in da.dims[:-2] if dim in da.coords}
    return xr.DataArray(blocks.map_blocks(warp_block, dtype=dtype), dims=da.dims,
                        coords={**coords, 'y': ys, 'x': xs}, name=da.name, attrs=da.attrs)


def _has_grid(layers):
    """True if the coordinates of a level of a product are finite (an incomplete GRS store has NaN ones)."""
    return all(layers.sizes[dim] > 1 and np.isfinite(layers[dim].values[[0, -1]]).all() for dim in ('x', 'y'))


class _AreaLevels:
    """
    Image of one acquisition over an area (sequence used by :class:`~grstbx.visual.Multiscale`): at each level
    of the pyramid, the same level of the products of its tiles, clipped to the area and combined on the grid
    of the level. All stays lazy: only the window displayed is read. The tiles of the UTM zone of the grid
    are aligned on it (exactly at full resolution; at the coarser levels, the pyramids of neighbouring tiles
    are offset by a fraction of a pixel, and their nearest pixel is taken); the tiles of another zone are
    reprojected block by block (nearest neighbour, see :func:`_warp`). Each level is built when first read.
    """

    def __init__(self, paths, grids, kind, params=None, mask_invalid=True):
        self.paths, self.grids, self.kind, self.params, self.mask_invalid = paths, grids, kind, params, mask_invalid
        self.grid = grids[0]
        self._products = {}
        self._images = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.grids)

    def resolution(self, level):
        """Pixel size of a level (see :meth:`grstbx.visual.Multiscale.resolution`)."""
        xs = self.grids[level]['x']
        return float(xs[1] - xs[0])

    def __getitem__(self, level):
        if level < 0:
            level += len(self)
        if not 0 <= level < len(self):
            raise IndexError(level)
        with self._lock:         # several threads read the dates of a time series
            if level not in self._images:
                self._images[level] = self._mosaic(level)
        return self._images[level]

    def _product(self, path):
        if path not in self._products:
            self._products[path] = _LazyLevels(path, self.kind, self.params, self.mask_invalid)
        return self._products[path]

    @staticmethod
    def _warp(layers, grid):
        if isinstance(layers, xr.Dataset):
            return xr.Dataset({name: _warp(layers[name], grid) for name in layers.data_vars})
        return _warp(layers, grid)

    def _piece(self, layers, grid):
        """Part of a product (one level) on the grid of a level of the area, None if it does not cover it."""
        import pyproj
        from rasterio.warp import transform_bounds

        crs, xs, ys = grid['crs'], grid['x'], grid['y']
        res = xs[1] - xs[0]
        if pyproj.CRS(layers.rio.crs) == crs:
            ydesc = bool(layers.y[0] > layers.y[-1])
            piece = layers.sel(x=slice(xs[0] - res, xs[-1] + res),
                               y=slice(ys[0] + res, ys[-1] - res) if ydesc else slice(ys[-1] - res, ys[0] + res))
            if piece.sizes['x'] == 0 or piece.sizes['y'] == 0:
                return None
            return piece.reindex(x=xs, y=ys, method='nearest', tolerance=res / 2)
        xmin, ymin, xmax, ymax = transform_bounds(crs, layers.rio.crs, xs[0], ys[-1], xs[-1], ys[0], densify_pts=21)
        lxmin, lymin, lxmax, lymax = layers.rio.bounds()
        if xmax <= lxmin or xmin >= lxmax or ymax <= lymin or ymin >= lymax:
            return None
        return self._warp(layers, grid)

    def _mosaic(self, level):
        grid = self.grids[level]
        image = layers = None
        for path in self.paths:
            product = self._product(path)
            # a product with fewer levels gives its coarsest one
            layers = product[min(level, len(product) - 1)]
            if not _has_grid(layers):
                if level == 0:
                    logger.warning(f'{path}: no coordinates (incomplete product?), left out of the area')
                continue
            piece = self._piece(layers, grid)
            if piece is None:
                continue
            # same blocks for all the pieces, so that their combination does not multiply them
            piece = piece.chunk({'x': AREA_BLOCK, 'y': AREA_BLOCK})
            # first valid pixel: the tiles of an acquisition overlap with the same values
            image = piece if image is None else image.combine_first(piece)
        if image is None:      # no product covers the area: empty image on the grid
            image = layers.isel(x=slice(0, 1), y=slice(0, 1)).assign_coords(
                x=grid['x'][:1], y=grid['y'][:1]).reindex(x=grid['x'], y=grid['y'])
            image = image.where(False)
        return image.drop_vars('spatial_ref', errors='ignore').rio.write_crs(grid['crs'])


def _open_area(products, area, kind, params, mask, first):
    """
    Images of the acquisitions of ``products`` over ``area`` (see :class:`_AreaLevels`): the products of
    the same satellite and sensing time are combined, on the grids of the UTM zone of most products.

    :param first: first product (opened), for its CRS when the products are not described
    :return: (images, times, grid of the full resolution)
    """
    import pyproj

    crss = [_crs_of(row) for row in products.itertuples()]
    known = [crs for crs in crss if crs is not None]
    crs = max(known, key=known.count) if known else pyproj.CRS(first.rio.crs)
    # pixel sizes, levels and alignment from a product of that zone, with a pyramid when there is one
    in_zone = [path for path, product_crs in zip(products.path, crss) if product_crs == crs] or [products.path[0]]
    candidates = [path for path in in_zone if is_zarr(path)] + [path for path in in_zone if not is_zarr(path)]
    template = None
    for path in candidates:
        template = _LazyLevels(path, kind, params, mask)
        if _has_grid(template[0]):
            break
        template = None
    if template is None:
        raise ValueError('no product of the area has coordinates (incomplete products?)')
    grids = _area_grids(area, crs, template)
    images, times = [], []
    for (_, time), group in products.groupby([products.satellite.fillna(''), products.time], sort=True):
        images.append(_AreaLevels(list(group.path), grids, kind, params, mask))
        times.append(time)
    order = np.argsort(times, kind='stable')
    return [images[i] for i in order], [times[i] for i in order], grids[0]


def open_products(paths, mask_invalid=True, min_size=512, tile=None, start=None, end=None,
                  last_processing=True, mask_l2a=True, cache_size=2e9, max_cloud_cover=None, area=None):
    """
    Open products for the viewers, lazily: each product (and each resolution level) is read when it is
    displayed, so that long time series open instantly.

    The products can be on different grids (e.g. subsets of several tiles), but must share the same CRS.

    :param paths: paths, glob patterns or folders, or a list of them (see :func:`catalogue`): L2A or L2B
                  products (one image each, all of the same level), folders of products (e.g. the tree
                  ``<tile>/YYYY/MM/DD/<product>`` of the GRS batches), or a single NetCDF datacube with a
                  ``time`` dimension
    :param mask_invalid: L2B: hide the pixels with ``mask != 0`` (not valid water pixels)
    :param min_size: size (pixels) of the coarsest level computed for the products without pyramid
    :param tile: tile to keep (needed when the products are in several UTM zones)
    :param start: first acquisition date (included)
    :param end: last acquisition date (included)
    :param last_processing: keep only the last processing of each acquisition (see :func:`select_products`)
    :param mask_l2a: L2A: hide the pixels with ``mask != 0`` (not valid water pixels; the products without
                     ``mask``, e.g. hGRS, are not masked)
    :param cache_size: memory (bytes) of the data kept in cache by the viewers (see
                       :class:`~grstbx.visual.Multiscale`)
    :param max_cloud_cover: maximum cloud cover (%) of the products with a STAC Item (see
                            :func:`select_products`)
    :param area: open only this area (shapely geometry or (lon_min, lat_min, lon_max, lat_max), lon/lat):
                 one image per acquisition, the products of its tiles clipped to the bounding box of the
                 area and combined on one grid per level of their pyramids, even across UTM zones; only the
                 window displayed is read (at most :data:`AREA_MAX_PIXELS_PYRAMID` pixels per side, or
                 :data:`AREA_MAX_PIXELS` for products without pyramid; a pin opens :data:`AREA_MIN_EXTENT` m
                 around it)
    :return: (data, kind, params, title): :class:`~grstbx.visual.Multiscale`, ``'L2A'`` or ``'L2B'``,
             displayed parameters (L2B, None for L2A), Markdown title of the viewer
    :raises SeveralZones: the products are in several UTM zones, and no tile nor area is given
    """
    from .visual import Multiscale

    items = expand_paths(paths)
    if not items:
        raise ValueError('no product given')

    # NetCDF datacube: one file with a time dimension
    if len(items) == 1 and not is_zarr(items[0]) and os.path.isfile(items[0]) and not is_hgrs(items[0]):
        cube = xr.open_dataset(items[0], decode_coords='all', chunks={})
        if 'time' in cube.dims and cube.sizes['time'] > 1:
            cube = cube.sortby('time')
            kind = 'L2A' if _is_l2a(cube) else 'L2B'
            params = None if kind == 'L2A' else _parameters(cube, dims=('time', 'y', 'x'))
            if kind == 'L2B' and not params:
                raise ValueError(f'no parameter to display in {items[0]}')
            layers = _layers(cube, kind, params, mask_l2a if kind == 'L2A' else mask_invalid)
            if layers.nbytes < MAX_IN_MEMORY:
                layers = layers.load()
            data = Multiscale.from_xarray(layers, min_size=min_size, cache_size=cache_size)
            data.scheduler = 'synchronous'   # dask scheduler of the exports (NetCDF)
            title = f'## {os.path.basename(items[0])}\n{kind} datacube, {cube.sizes["time"]} dates'
            return data, kind, params, title

    products = select_products(catalogue(items), tile=tile, start=start, end=end,
                               last_processing=last_processing, max_cloud_cover=max_cloud_cover)
    if products.empty:
        raise ValueError('no product found (check the folders, the tile and the period)')
    if products.level.nunique() > 1:
        # the level of the names can be wrong (e.g. L2B products named MSIL2Agrs): check the content of the
        # first product of each level
        content = {level: 'L2A' if _is_l2a(_open_level(group.path.iloc[0])) else 'L2B'
                   for level, group in products.groupby('level')}
        if len(set(content.values())) > 1:
            counts = products.level.value_counts().to_dict()
            raise ValueError(f'L2A and L2B products cannot be displayed together ({counts}): open the products '
                             'of one level')
    tiles = products.tile.dropna()
    if area is None and tiles.map(_utm_zone).nunique() > 1:
        raise SeveralZones(tiles.value_counts().sort_index().to_dict())

    # level and parameters from the first product, the others are opened when displayed
    first = _open_level(products.path[0])
    kind = 'L2A' if _is_l2a(first) else 'L2B'
    params = None if kind == 'L2A' else _parameters(first)
    if kind == 'L2B' and not params:
        raise ValueError(f'no parameter to display in {products.path[0]}')
    mask = mask_l2a if kind == 'L2A' else mask_invalid
    if area is not None:
        from shapely.geometry import box

        area = box(*area) if isinstance(area, (list, tuple)) else area
        images, times, grid = _open_area(products, area, kind, params, mask, first)
        data = Multiscale(images, np.array(times, dtype='datetime64[ns]'), min_size=min_size, cache_size=cache_size)
        data.scheduler = 'synchronous'    # the reprojected blocks read the products themselves
        dates = pd.DatetimeIndex(data.times)
        where = f', tiles {", ".join(sorted(tiles.unique()))}' if len(tiles) else ''
        title = (f'## {len(images)} {kind} images of the area\n{len(products)} products{where}, '
                 f'{len(grid["x"])} x {len(grid["y"])} pixels, from {dates[0].date()} to {dates[-1].date()}')
        return data, kind, params, title
    images = [_LazyLevels(path, kind, params, mask) for path in products.path]
    images[0]._levels[0] = _layers(first, kind, params, mask)
    times = [time if not pd.isna(time) else pd.Timestamp(_image_time(images[i][0], products.path[i]))
             for i, time in enumerate(products.time)]
    data = Multiscale(images, np.array(times, dtype='datetime64[ns]'), min_size=min_size, cache_size=cache_size)
    data.scheduler = None if all(is_zarr(path) for path in products.path) else 'synchronous'

    if len(products) == 1:
        level = 'hGRS L2A' if is_hgrs(products.path[0]) else f'GRS {kind}'
        title = f'## {os.path.basename(products.path[0])}\n{level}'
    else:
        dates = pd.DatetimeIndex(data.times)
        where = f', tile {", ".join(sorted(tiles.unique()))}' if len(tiles) else ''
        title = f'## {len(products)} {kind} products{where}\nfrom {dates[0].date()} to {dates[-1].date()}'
    return data, kind, params, title


# -- exports ------------------------------------------------------------------------------------------

def _is_spectral(viewer):
    return hasattr(viewer, 'wls')


def _as_layers(obj, viewer):
    """(labels, DataArray with a 'layer' first dimension) of an image or pixel of the viewer."""
    if _is_spectral(viewer):
        da = obj.sel(wl=list(viewer.wls)).rename(wl='layer')
        return [f'Rrs_{wl:g}' for wl in viewer.wls], da
    return list(viewer.params), obj[list(viewer.params)].to_array('layer')


def _compute(objects, viewer):
    """
    Compute lazy xarray objects together, in a single dask graph. NetCDF (HDF5) reads are serialized by a
    lock: they are faster on a single thread (see :class:`~grstbx.driver.L2grs`).
    """
    import dask

    with warnings.catch_warnings():   # all-NaN pixels or polygons
        warnings.simplefilter('ignore', RuntimeWarning)
        return dask.compute(*objects, scheduler=getattr(viewer.data, 'scheduler', None))


def points_values(viewer):
    """
    Full-resolution values of the points of the map, for all the dates of the viewer: spectra
    (:class:`~grstbx.visual.ViewSpectral`) or parameters (:class:`~grstbx.visual.ViewParam`).

    :return: pandas.DataFrame, one row per point and date, one column per wavelength or parameter
    """
    points = viewer.get_points()
    native = points.to_crs(viewer.data.crs)
    lazy, rows = [], []
    for ipoint, (point, xy) in enumerate(zip(points.geometry, native.geometry)):
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            xmin, ymin, xmax, ymax = viewer.data.bounds(idate)
            if not (xmin <= xy.x <= xmax and ymin <= xy.y <= ymax):   # point outside the image
                continue
            pixel = viewer.data.images[idate][0].sel(x=xy.x, y=xy.y, method='nearest')
            labels, layers = _as_layers(pixel, viewer)
            lazy.append(layers)
            rows.append({'point': ipoint, 'lon': point.x, 'lat': point.y, 'time': date})
    for row, values in zip(rows, _compute(lazy, viewer)):
        row.update({label: float(v) for label, v in zip(labels, values.values)})
    return pd.DataFrame(rows)



def polygons_statistics(viewer):
    """
    Statistics of the full-resolution valid pixels within each polygon of the map, for all the dates of
    the viewer: Rrs (:class:`~grstbx.visual.ViewSpectral`) or parameters (:class:`~grstbx.visual.ViewParam`).

    :return: pandas.DataFrame, one row per polygon, date and statistic (median, mean, std, q05, q25, q75,
             q95, count), one column per wavelength or parameter
    """
    from rasterio.features import geometry_mask

    n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
    lazy, keys, masks = [], [], {}
    for ipoly in range(n_poly):
        aoi = viewer.get_geom(crs=viewer.data.crs, index=ipoly)
        xmin, ymin, xmax, ymax = aoi.total_bounds
        for date, idate in zip(viewer.datetimes, viewer.indexes):
            image = viewer.data.images[idate][0]
            ys = image.y.values
            # bounding box of the polygon (lazy slicing)
            image = image.sel(x=slice(xmin, xmax), y=slice(ymax, ymin) if ys[0] > ys[-1] else slice(ymin, ymax))
            if image.sizes['x'] == 0 or image.sizes['y'] == 0:   # polygon outside the image
                continue
            # pixels inside the polygon; the mask is computed once per grid (e.g. for all the dates of a
            # datacube)
            grid = (ipoly, image.sizes['y'], image.sizes['x'], tuple(image.rio.transform()))
            if grid not in masks:
                masks[grid] = xr.DataArray(
                    geometry_mask(aoi.geometry.values, out_shape=grid[1:3], transform=image.rio.transform(),
                                  invert=True), dims=('y', 'x'), coords={'y': image.y, 'x': image.x})
            labels, layers = _as_layers(image.where(masks[grid]), viewer)
            lazy.append(layers)
            keys.append((ipoly, date))

    rows = []
    for (ipoly, date), layers in zip(keys, _compute(lazy, viewer)):
        values = layers.values.reshape(layers.shape[0], -1)
        values = values[:, np.isfinite(values).any(axis=0)]
        if values.shape[1] == 0:   # no valid pixel
            continue
        with warnings.catch_warnings():   # all-NaN layers
            warnings.simplefilter('ignore', RuntimeWarning)
            stats = {'median': np.nanmedian(values, axis=1), 'mean': np.nanmean(values, axis=1),
                     'std': np.nanstd(values, axis=1), 'count': np.isfinite(values).sum(axis=1)}
            for q in (0.05, 0.25, 0.75, 0.95):
                stats[f'q{q * 100:02.0f}'] = np.nanquantile(values, q, axis=1)
        for stat, column in stats.items():
            row = {'polygon': ipoly, 'time': date, 'statistic': stat}
            row.update({label: float(v) for label, v in zip(labels, column)})
            rows.append(row)
    return pd.DataFrame(rows)


def _csv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def _geojson(gdf):
    return io.StringIO(gdf.to_json())


def _exports(viewer):
    """Download buttons of the points, polygons, values and statistics drawn on the map."""
    import geopandas as gpd

    def polygons():
        n_poly = len((viewer.aoi_stream.data or {}).get('xs', []))
        if n_poly == 0:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        return pd.concat([viewer.get_geom(index=i) for i in range(n_poly)], ignore_index=True)

    values = 'Spectra' if _is_spectral(viewer) else 'Parameters'
    buttons = [
        pn.widgets.FileDownload(callback=lambda: _csv(points_values(viewer)), filename='points_values.csv',
                                label=f'{values} of the points (CSV)', button_type='primary'),
        pn.widgets.FileDownload(callback=lambda: _csv(polygons_statistics(viewer)),
                                filename='polygons_statistics.csv',
                                label='Statistics of the polygons (CSV)', button_type='primary'),
        pn.widgets.FileDownload(callback=lambda: _geojson(viewer.get_points()), filename='points.geojson',
                                label='Points (GeoJSON)'),
        pn.widgets.FileDownload(callback=lambda: _geojson(polygons()), filename='polygons.geojson',
                                label='Polygons (GeoJSON)'),
    ]
    return pn.Column(
        '### Export',
        f'Full-resolution {values.lower()} of the points and statistics of the polygons drawn on the map, for '
        f'all the dates ({len(viewer.indexes)}). The polygon statistics read all the pixels of the polygons: '
        'large polygons or long time series take a while.',
        pn.Row(*buttons))


# -- dashboard ---------------------------------------------------------------------------------------

def _file_browser(root, products_input):
    """
    Compact file browser of the server for the sidebar: a click on a folder opens it, a click on a
    product (see :func:`is_product`) adds it to ``products_input``.
    """
    folder = pn.widgets.TextInput(name='Folder', value=str(root))
    entries = pn.widgets.MultiSelect(options={}, size=16, sizing_mode='stretch_width')
    parent = pn.widgets.Button(name='Parent folder', icon='arrow-up', button_type='light')
    add_all = pn.widgets.Button(name='Add all the products', icon='list-check', button_type='light')
    add_folder = pn.widgets.Button(name='Add this folder (time series)', icon='folder-plus', button_type='light')

    def listing():
        try:
            names = sorted(n for n in os.listdir(folder.value) if not n.startswith('.'))
        except OSError as exc:
            pn.state.notifications.error(str(exc))
            return [], []
        paths = [os.path.join(folder.value, n) for n in names]
        products = [p for p in paths if is_product(p)]
        return [p for p in paths if os.path.isdir(p) and p not in products], products

    def refresh(*events):
        folders, products = listing()
        options = {'\N{FILE FOLDER} ' + os.path.basename(p): p for p in folders}
        options.update({'\N{GLOBE WITH MERIDIANS} ' + os.path.basename(p): p for p in products})
        entries.options = options

    def add(paths):
        current = [line.strip() for line in products_input.value.splitlines() if line.strip()]
        products_input.value = '\n'.join(current + [p for p in paths if p not in current])

    def on_click(event):
        if not event.new:
            return
        selected = event.new[0]
        entries.value = []
        if is_product(selected):
            add([selected])
        else:
            folder.value = selected

    def on_parent(event):
        folder.value = os.path.dirname(folder.value.rstrip('/')) or '/'

    folder.param.watch(refresh, 'value')
    entries.param.watch(on_click, 'value')
    parent.on_click(on_parent)
    add_all.on_click(lambda event: add(listing()[1]))
    add_folder.on_click(lambda event: add([folder.value.rstrip('/')]))
    refresh()
    return pn.Column(folder, pn.Row(parent, add_all), add_folder, entries,
                     pn.pane.Markdown('\N{GLOBE WITH MERIDIANS}: product (click to add it), '
                                      '\N{FILE FOLDER}: folder (click to open it). *Add this folder* adds all '
                                      'the products below it, e.g. a tile folder `<tile>/YYYY/MM/DD/<product>`.'),
                     sizing_mode='stretch_width')


#: area shown by the search map at start-up (lon_min, lat_min, lon_max, lat_max): the Etang de Berre and the
#: Gulf of Fos
SEARCH_AREA = (4.80, 43.30, 5.40, 43.62)


class SearchMap:
    """
    Empty map of the area of a STAC search, over a basemap: pins (Point Draw tool), polygons (Polygon Draw
    tool, double-click to start and to close) and lasso shapes (Freehand Draw tool, closed between their
    first and last points). The area of the search is the union of the shapes drawn (see :meth:`geometry`).

    :param basemap: basemap (see :data:`grstbx.visual.BASEMAPS`)
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :param area: area shown at start-up (lon_min, lat_min, lon_max, lat_max), default :data:`SEARCH_AREA`;
                 None: the whole world
    """

    # half width of the web Mercator world (m)
    WORLD = 20037508.34

    def __init__(self, basemap=None, width=1000, height=450, area=SEARCH_AREA):
        from . import visual

        self.basemap = basemap or visual.DEFAULT_BASEMAP
        self.width, self.height, self.area = width, height, area
        self.pane = pn.pane.HoloViews()
        self._subscribers = []
        self.clear()

    def clear(self, event=None):
        """Remove the shapes drawn: new elements and drawing streams."""
        import holoviews as hv

        from . import visual

        pins = hv.Points([]).opts(color='#c0392b', size=10, line_color='white')
        polygons = hv.Polygons([]).opts(fill_color='#1f4e79', fill_alpha=0.2, line_color='#1f4e79', line_width=2)
        lassos = hv.Path([]).opts(line_color='#1f4e79', line_width=2)
        self.pin_stream = hv.streams.PointDraw(source=pins)
        self.polygon_stream = hv.streams.PolyDraw(source=polygons, drag=True, show_vertices=True)
        self.lasso_stream = hv.streams.FreehandDraw(source=lassos)
        for stream in self.streams:
            for subscriber in self._subscribers:
                stream.add_subscriber(subscriber)
        xlim, ylim = self.extent()
        tiles = visual.basemap_tiles(self.basemap).opts(
            width=self.width, height=self.height, xlim=xlim, ylim=ylim, xaxis=None, yaxis=None)
        self.pane.object = tiles * polygons * lassos * pins

    def extent(self):
        """
        (xlim, ylim) of the start-up view in web Mercator: the area, enlarged to the shape of the map so that
        the tiles stay square; the whole world without area.
        """
        if self.area is None:     # whole world in longitude, centred on 30 deg N
            half_width, centre = self.WORLD, (0., 3.5e6)
        else:
            import pyproj

            to_map = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
            (x0, y0), (x1, y1) = to_map.transform(*self.area[:2]), to_map.transform(*self.area[2:])
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            half_width = max(x1 - x0, (y1 - y0) * self.width / self.height) / 2
        half_height = half_width * self.height / self.width
        return ((centre[0] - half_width, centre[0] + half_width),
                (centre[1] - half_height, centre[1] + half_height))

    @property
    def streams(self):
        return self.pin_stream, self.polygon_stream, self.lasso_stream

    def on_draw(self, callback):
        """Call ``callback()`` each time a shape is drawn or edited."""
        def subscriber(**kwargs):
            callback()

        self._subscribers.append(subscriber)
        for stream in self.streams:
            stream.add_subscriber(subscriber)

    def geometry(self):
        """Union of the shapes drawn (shapely geometry, lon/lat), None if nothing is drawn."""
        import pyproj
        import shapely
        from shapely.geometry import Point, Polygon

        to_lonlat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
        shapes = []
        pins = self.pin_stream.data or {}
        shapes += [Point(to_lonlat.transform(x, y)) for x, y in zip(pins.get('x', []), pins.get('y', []))]
        for stream in (self.polygon_stream, self.lasso_stream):
            data = stream.data or {}
            for xs, ys in zip(data.get('xs', []), data.get('ys', [])):
                if len(xs) >= 3:      # unfinished polygons are left out
                    lon, lat = to_lonlat.transform(np.asarray(xs, float), np.asarray(ys, float))
                    # a lasso can cross itself
                    shapes.append(shapely.make_valid(Polygon(zip(lon, lat))))
        return shapely.union_all(shapes) if shapes else None


def create_app(path=None, root=None, reproject=True, basemap=None, cmap=None, width=820, height=620,
               tile=None, start=None, end=None, last_processing=True, mask_l2a=True, mask_l2b=True,
               cache_size=2e9, max_cloud_cover=None, stac=None, search_area=SEARCH_AREA):
    """
    Build the dashboard (one per browser session).

    :param path: product(s) opened at start-up (optional): path, glob pattern or list of them
    :param root: directory of the file browser of the sidebar (default: directory of the first product or
                 current directory)
    :param reproject: display over basemaps (web Mercator)
    :param basemap: initial basemap (see :data:`grstbx.visual.BASEMAPS`)
    :param cmap: initial colormap (see :class:`~grstbx.visual.ViewSpectral`)
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :param tile: tile of the time series (folders with several UTM zones)
    :param start: first acquisition date of the time series
    :param end: last acquisition date of the time series
    :param last_processing: keep only the last processing of each acquisition
    :param mask_l2a: hide the invalid pixels (``mask != 0``) of the L2A products
    :param mask_l2b: hide the invalid pixels (``mask != 0``) of the L2B products
    :param cache_size: memory (bytes) of the data kept in cache by each session (see
                       :class:`~grstbx.visual.Multiscale`)
    :param max_cloud_cover: maximum cloud cover (%) of the products with a STAC Item
    :param stac: catalogue of the *STAC search* card (STAC API URL, or folder / glob pattern of STAC Items)
    :param search_area: area shown by the search map at start-up (lon_min, lat_min, lon_max, lat_max), default
                        the Etang de Berre (:data:`SEARCH_AREA`); None: the whole world
    :return: panel template
    """
    from . import visual

    pn.extension(notifications=True)
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path or [])
    if root is None:
        first = str(paths[0]).rstrip('/') if paths else ''
        root = os.path.dirname(first.split('*')[0]) if first else os.getcwd()
        if not os.path.isdir(root):
            root = os.getcwd()

    products_input = pn.widgets.TextAreaInput(
        name='Products', value='\n'.join(map(str, paths)), height=110, sizing_mode='stretch_width',
        placeholder='One product, folder or glob pattern per line, e.g.\n/data/L2A/product.zarr\n'
                    '/data/L2A_zarr/31TEJ\n/data/L2B/BEFR/2024/*/*/*.nc')
    browser = _file_browser(root, products_input)
    clear_button = pn.widgets.Button(name='Clear', button_type='light')
    reproject_box = pn.widgets.Checkbox(name='Basemap (reproject to web Mercator)', value=reproject)
    basemap_select = pn.widgets.Select(name='Basemap', options=list(visual.BASEMAPS),
                                       value=basemap or visual.DEFAULT_BASEMAP)
    cmap_select = pn.widgets.Select(name='Colormap', options=list(visual.RRS_COLORMAPS),
                                    value=next((label for label, name in visual.RRS_COLORMAPS.items()
                                                if cmap in (label, name)), list(visual.RRS_COLORMAPS)[0]))
    rrs_range = pn.widgets.EditableRangeSlider(name='L2A: initial Rrs range (sr-1)', start=0, end=0.1,
                                               value=(0, 0.02), step=0.001)
    param_range = pn.widgets.EditableRangeSlider(name='L2B: initial range of the parameters', start=0, end=100,
                                                 value=(0, 10), step=0.1)
    mask_l2a_box = pn.widgets.Checkbox(name='L2A: hide the invalid pixels (mask != 0)', value=mask_l2a)
    mask_box = pn.widgets.Checkbox(name='L2B: hide the invalid pixels (mask != 0)', value=mask_l2b)
    load_button = pn.widgets.Button(name='Open', button_type='primary')
    # time series of folders: tile, period and processings (see select_products)
    scan_button = pn.widgets.Button(name='Scan the products', icon='search', button_type='light')
    scan_summary = pn.pane.Markdown('Scan the folders of the *Products* field to list their tiles and dates.',
                                    sizing_mode='stretch_width')
    tile_select = pn.widgets.Select(name='Tile', options={'All the tiles': ''} | ({tile: tile} if tile else {}),
                                    value=tile or '')
    start_picker = pn.widgets.DatePicker(name='From', value=pd.Timestamp(start).date() if start else None,
                                         width=160)
    end_picker = pn.widgets.DatePicker(name='To', value=pd.Timestamp(end).date() if end else None, width=160)
    last_box = pn.widgets.Checkbox(name='Keep the last processing of each acquisition', value=last_processing)
    cloud_input = pn.widgets.FloatInput(name='Maximum cloud cover (%, products with a STAC Item)',
                                        value=max_cloud_cover, start=0, end=100, step=5, placeholder='no limit')
    series_card = pn.Card(scan_button, scan_summary, tile_select, pn.Row(start_picker, end_picker), last_box,
                          cloud_input,
                          title='Time series (folders)', collapsed=True, sizing_mode='stretch_width')
    # STAC search: local Items or STAC API (see search_items)
    stac_source = pn.widgets.TextInput(name='Catalogue: STAC API URL, or folder / glob pattern of Items',
                                       value=stac or '', sizing_mode='stretch_width',
                                       placeholder='/data/L2B or https://earth-search.aws.element84.com/v1')
    stac_collections = pn.widgets.TextInput(name='Collections (comma-separated)', value='',
                                            sizing_mode='stretch_width', placeholder='e.g. grs-l2b, sentinel-2-l2a')
    stac_area = pn.widgets.RadioButtonGroup(
        options=['Everywhere', 'Search map', 'Last polygon on the map', 'Bounding box'], value='Everywhere',
        button_type='light', orientation='vertical', sizing_mode='stretch_width')
    stac_bbox = pn.widgets.TextInput(name='Bounding box: lon_min, lat_min, lon_max, lat_max (deg)', value='',
                                     placeholder='4.4, 43.4, 4.7, 43.6', sizing_mode='stretch_width')
    stac_start = pn.widgets.DatePicker(name='From', width=160)
    stac_end = pn.widgets.DatePicker(name='To', width=160)
    stac_cloud = pn.widgets.FloatInput(name='Maximum cloud cover (%)', value=None, start=0, end=100, step=5,
                                       placeholder='no limit')
    stac_button = pn.widgets.Button(name='Search', icon='search', button_type='light')
    stac_summary = pn.pane.Markdown('Search the STAC Items of a catalogue by area, period and cloud cover.',
                                    sizing_mode='stretch_width')
    stac_clip = pn.widgets.Checkbox(name='Open the area only (tiles clipped and combined)', value=True,
                                    visible=False)
    stac_level = pn.widgets.Select(name='Level to open', options=[], visible=False, sizing_mode='stretch_width')
    stac_open = pn.widgets.Button(name='Open the results as a time series', icon='timeline', button_type='primary',
                                  disabled=True)
    stac_results = {}
    # empty map of the area of the search, below the viewer (see SearchMap)
    search_map = SearchMap(basemap=basemap_select.value, width=width, height=min(height, 520), area=search_area)
    search_map.on_draw(lambda: setattr(stac_area, 'value', 'Search map')
                       if search_map.geometry() is not None else None)
    clear_area = pn.widgets.Button(name='Clear the area', icon='eraser', button_type='light')
    clear_area.on_click(search_map.clear)
    area_card = pn.Card(
        pn.pane.Markdown('Draw the area of the **STAC search** (*Search map* in the sidebar): pins with the '
                         '**Point Draw** tool, polygons with the **Polygon Draw** tool (double-click to start and to '
                         'close) or lasso shapes with the **Freehand Draw** tool. The search keeps the products '
                         'that intersect one of the shapes.', sizing_mode='stretch_width'),
        search_map.pane, clear_area, title='Area of the STAC search', sizing_mode='stretch_width')
    stac_card = pn.Card(stac_source, stac_collections, stac_area, stac_bbox, pn.Row(stac_start, stac_end),
                        stac_cloud, stac_button, stac_summary, stac_level, stac_clip, stac_open,
                        title='STAC search', collapsed=not stac, sizing_mode='stretch_width')
    state = {'viewer': None}
    # controls of the opened viewer (filled when the products are opened)
    contrast_card = pn.Card(title='Contrast and brightness', visible=False, sizing_mode='stretch_width')
    rgb_card = pn.Card(title='RGB composite (L2A)', visible=False, sizing_mode='stretch_width')

    main = pn.Column(sizing_mode='stretch_width')
    intro = pn.pane.Markdown(
        '## Open GRS products\n\n'
        'Enter the products in the **Products** field of the sidebar, or add them with the **File browser**, '
        'then click **Open**. Or find them in a STAC catalogue: draw the area on the map below, then search in '
        'the **STAC search** card.', sizing_mode='stretch_width')
    help_text = pn.pane.Markdown(
        f'Enter the paths of {ACCEPTED} on the server, one path or glob pattern per line, or add them with '
        'the file browser, then click **Open**. Several products make a time series: they can be on different '
        'grids, but must be in the same CRS (UTM zone).\n\n'
        'A **folder** (e.g. a tile folder `<tile>/YYYY/MM/DD/<product>` of the GRS batches) adds all the products '
        'below it. The **Time series** card selects the tile (needed for several UTM zones), the period and '
        'the last processing of each acquisition; the products are read only when they are displayed, so '
        'that long time series open instantly.\n\n'
        '- **L2A** products are displayed with the spectral viewer: pick the band or the true-colour '
        'composite, hover the map to plot the full-resolution spectrum under the mouse (dashed line), '
        'add points with the **Point Draw** tool to plot their spectra;\n'
        '- **L2B** products are displayed with the parameter viewer: pick the parameter and the date, hover '
        'the map to plot the time series under the mouse, add points to plot their time series.\n\n'
        'Pan and zoom the map: only the pixels displayed are read, from the Zarr pyramid levels. Draw '
        'polygons with the **Polygon Draw** tool (double-click to start and to close), and download the '
        'values of the points and the statistics of the polygons below the map.\n\n'
        'With an area drawn on the search map, **Open the area only** of the STAC search card opens the area '
        'alone, the tiles of each acquisition combined.', sizing_mode='stretch_width')
    help_card = pn.Card(help_text, title='How to use the viewer', collapsed=True, sizing_mode='stretch_width')
    # start page: the search map first, the full help folded
    start_page = [intro, area_card, help_card]
    main.extend(start_page)

    def scan(event=None):
        """List the products of the Products field: tiles, levels, number of products and period."""
        try:
            products = catalogue(products_input.value)
        except Exception as exc:
            scan_summary.object = f'Cannot scan the products: {exc}'
            return None
        if products.empty:
            scan_summary.object = 'No product found.'
            return products
        rows = ['| tile | level | products | last processing | from | to |', '|---|---|---|---|---|---|']
        options = {'All the tiles': ''}
        for (tile_, level), group in products.fillna({'tile': '?'}).groupby(['tile', 'level']):
            last = select_products(group)
            rows.append(f'| {tile_} | {level} | {len(group)} | {len(last)} | {group.time.min():%Y-%m-%d} | '
                        f'{group.time.max():%Y-%m-%d} |')
            if tile_ != '?':
                options[f'{tile_} ({level}, {len(last)} product{"s" if len(last) > 1 else ""})'] = tile_
        scan_summary.object = '\n'.join(rows)
        current = tile_select.value
        tile_select.options = options
        tile_select.value = current if current in options.values() else ''
        for picker in (start_picker, end_picker):
            picker.start, picker.end = products.time.min().date(), products.time.max().date()
        return products

    def show_failed():
        """After a failed opening: the viewer already open stays, with its polygons and points."""
        if state['viewer'] is None:
            area_card.collapsed = False
            main.objects = start_page
            contrast_card.visible = rgb_card.visible = False

    def on_load(event, area=None):
        """Open the products of the Products field (``area``: only this area, see open_products)."""
        if not products_input.value.strip():
            pn.state.notifications.warning('Enter or add products first.')
            return
        load_button.loading = True
        main.loading = True
        try:
            data, kind, params, title = open_products(
                products_input.value, mask_invalid=mask_box.value, mask_l2a=mask_l2a_box.value,
                tile=tile_select.value or None, cache_size=cache_size,
                start=start_picker.value, end=end_picker.value, last_processing=last_box.value,
                max_cloud_cover=cloud_input.value, area=area)
            ndates = len(data.images)
            # the live time series of L2B reads one block per date: off at opening for long series
            live = kind == 'L2A' or ndates <= LIVE_MAX_DATES
            options = dict(reproject=reproject_box.value, basemap=basemap_select.value, cmap=cmap_select.value,
                           width=width, height=height, title=title, live=live)
            if kind == 'L2A':
                viewer = visual.ViewSpectral(data, minmaxvalues=tuple(rrs_range.value),
                                             minmax=(0, max(0.1, rrs_range.value[1])), **options)
            else:
                viewer = visual.ViewParam(data, params=params, minmaxvalues=tuple(param_range.value),
                                          minmax=(0, max(100, param_range.value[1])), **options)
            if not live:
                pn.state.notifications.info(
                    f'{ndates} dates: the live time series under the mouse is off (it reads every date); tick '
                    '"Live time series" to turn it on, or draw points.', duration=10000)
            state['viewer'] = viewer
            # the search map stays below the viewer, folded
            area_card.collapsed = True
            main.objects = [viewer.visu(controls=False), _exports(viewer), area_card]
            contrast_card.objects = [viewer.display_controls()]
            contrast_card.visible = True
            rgb_card.objects = [viewer.rgb_controls()] if kind == 'L2A' else []
            rgb_card.visible = kind == 'L2A'
        except SeveralZones as exc:
            scan()
            series_card.collapsed = False
            pn.state.notifications.warning(f'{exc}: choose it in the "Time series" card, then click Open.',
                                           duration=0)
            show_failed()
        except Exception as exc:
            pn.state.notifications.error(f'Cannot open the products: {exc}', duration=0)
            show_failed()
        finally:
            load_button.loading = False
            main.loading = False

    def stac_search(event=None):
        """Search the catalogue of the STAC card and list the matching products."""
        stac_open.disabled = True
        stac_level.visible = False
        stac_results.clear()
        if not stac_source.value.strip():
            pn.state.notifications.warning('Enter a STAC API URL or a folder of STAC Items first.')
            return
        area = {}
        try:
            if stac_area.value == 'Bounding box':
                bbox = [float(v) for v in stac_bbox.value.replace(';', ',').split(',')]
                if len(bbox) != 4:
                    raise ValueError('the bounding box needs 4 values: lon_min, lat_min, lon_max, lat_max')
                area['bbox'] = bbox
            elif stac_area.value == 'Search map':
                area['intersects'] = search_map.geometry()
                if area['intersects'] is None:
                    area_card.collapsed = False
                    raise ValueError('draw a pin, a polygon or a lasso on the search map first')
            elif stac_area.value == 'Last polygon on the map':
                viewer = state['viewer']
                try:
                    polygon = viewer.get_geom() if viewer is not None else None
                except (IndexError, KeyError, TypeError, ValueError):      # no polygon drawn
                    polygon = None
                if polygon is None or polygon.empty:
                    raise ValueError('open products and draw a polygon on the map first (Polygon Draw tool)')
                area['intersects'] = polygon.geometry.iloc[0]
            collections = [c.strip() for c in stac_collections.value.split(',') if c.strip()] or None
            stac_button.loading = True
            found = search_items(stac_source.value.strip(), start=stac_start.value, end=stac_end.value,
                                 max_cloud_cover=stac_cloud.value, collections=collections, **area)
        except Exception as exc:
            stac_summary.object = f'Search failed: {exc}'
            return
        finally:
            stac_button.loading = False
        if found.empty:
            stac_summary.object = 'No STAC Item matches the search.'
            return
        rows = ['| date | tile | cloud (%) | product |', '|---|---|---|---|']
        for _, row in found.head(15).iterrows():
            cloud = '' if pd.isna(row.cloud_cover) else f'{row.cloud_cover:.1f}'
            rows.append(f'| {row.time:%Y-%m-%d} | {row.tile or "?"} | {cloud} | {row["name"]} |')
        more = f'\n\n... and {len(found) - 15} more' if len(found) > 15 else ''
        tiles = ', '.join(f'{t} ({n})' for t, n in found.tile.fillna('?').value_counts().sort_index().items())
        counts = found.level.value_counts().sort_index()
        levels = ', '.join(f'{level} ({n})' for level, n in counts.items())
        stac_summary.object = (f'**{len(found)} products** from {found.time.min():%Y-%m-%d} to '
                               f'{found.time.max():%Y-%m-%d}, tiles {tiles}, levels {levels}\n\n'
                               + '\n'.join(rows) + more)
        # the products of a time series have the same level: choose it when the results mix them
        stac_level.options = list(counts.index)
        stac_level.value = counts.index[-1]
        stac_level.visible = len(counts) > 1
        # the Items (local files, or URLs of the STAC API Items), described again at opening; the data
        # assets for the Items without location
        stac_results['paths'] = [item if os.path.isfile(str(item)) or '://' in str(item)
                                 else path for item, path in zip(found.item, found.path)]
        stac_results['levels'] = list(found.level)
        # area opened with "Open the area only" (lon/lat)
        stac_results['area'] = area.get('intersects', area.get('bbox'))
        stac_clip.visible = stac_results['area'] is not None
        stac_open.disabled = False

    def stac_to_series(event):
        paths, levels = stac_results.get('paths', []), stac_results.get('levels', [])
        if stac_level.visible:
            paths = [path for path, level in zip(paths, levels) if level == stac_level.value]
        products_input.value = '\n'.join(paths)
        on_load(None, area=stac_results.get('area') if stac_clip.value else None)

    load_button.on_click(on_load)
    scan_button.on_click(scan)
    stac_button.on_click(stac_search)
    stac_open.on_click(stac_to_series)
    clear_button.on_click(lambda event: setattr(products_input, 'value', ''))
    if paths:
        pn.state.onload(lambda: on_load(None))

    sidebar = [
        products_input,
        pn.Row(load_button, clear_button),
        pn.Card(browser, title='File browser', collapsed=bool(paths), sizing_mode='stretch_width'),
        series_card,
        stac_card,
        pn.Card(reproject_box, basemap_select, cmap_select, rrs_range, param_range, mask_l2a_box, mask_box,
                title='Options (applied at opening)', collapsed=True, sizing_mode='stretch_width'),
        contrast_card,
        rgb_card,
    ]
    return pn.template.FastListTemplate(title='grstbx viewer', sidebar=sidebar, main=[main],
                                        sidebar_width=380, header_background='#1f4e79')


def panel(path=None, root=None, sidebar_width=380, **kwargs):
    """
    Dashboard displayed inline in a Jupyter notebook: the sidebar and the map of :func:`create_app`, in a
    row (the Panel templates are not rendered in the notebook cells), served by the kernel.

    .. code-block:: python

        from grstbx import app
        app.panel('/data/L2B/BEFR/2024/*/*/*.nc')

    :param path: product(s) opened at start-up (optional): path, glob pattern or list of them
    :param root: directory of the file browser
    :param sidebar_width: width of the sidebar (pixels)
    :param kwargs: other arguments of :func:`create_app`
    :return: panel layout
    """
    template = create_app(path=path, root=root, **kwargs)
    return pn.Row(pn.Column(*template.sidebar, width=sidebar_width), *template.main)


def serve(path=None, root=None, port=5006, address=None, show=True, allow_websocket_origin=None, **kwargs):
    """
    Serve the dashboard (blocking).

    :param path: product(s) opened at start-up (optional): path, glob pattern or list of them
    :param root: directory of the file browser
    :param port: port of the server
    :param address: address to listen on (e.g. '0.0.0.0' to accept connections from other computers)
    :param show: open the dashboard in the web browser
    :param allow_websocket_origin: list of hosts allowed to connect (e.g. ['myserver:5006']), added to the
                                   names of the local machine (localhost, 127.0.0.1, host name, address)
    :param kwargs: other arguments of :func:`create_app`
    """
    pn.serve(lambda: create_app(path=path, root=root, **kwargs), port=port, address=address, show=show,
             websocket_origin=_websocket_origins(port, address, allow_websocket_origin),
             title='grstbx viewer')


def _websocket_origins(port, address=None, extra=None):
    """
    Hosts allowed to open the websocket of the dashboard. Bokeh allows only ``localhost:<port>`` by default,
    so that a page opened as ``127.0.0.1:<port>`` (or with the host name) stays empty.
    """
    import socket

    hosts = {'localhost', '127.0.0.1', socket.gethostname(), socket.getfqdn()}
    if address and address not in ('0.0.0.0', '::'):
        hosts.add(address)
    origins = sorted(f'{host}:{port}' for host in hosts if host)
    return origins + [origin for origin in (extra or []) if origin not in origins]


def main(argv=None):
    """Command line entry point: ``grstbx-viewer``."""
    parser = argparse.ArgumentParser(prog='grstbx-viewer',
                                     description='Web dashboard to explore GRS L2A and L2B products, '
                                                 'single images or time series.')
    parser.add_argument('path', nargs='*',
                        help=f'products opened at start-up (paths, folders or quoted glob patterns): {ACCEPTED}')
    parser.add_argument('--root', help='directory of the file browser (default: directory of the first '
                                       'product, or current directory)')
    parser.add_argument('--port', type=int, default=5006, help='port of the server (default: 5006)')
    parser.add_argument('--address', help="address to listen on, e.g. 0.0.0.0 to accept connections from other "
                                          "computers (default: localhost)")
    parser.add_argument('--allow-websocket-origin', action='append',
                        help='host[:port] allowed to connect, repeat for several hosts; localhost, 127.0.0.1 and '
                             'the host name of the machine are always allowed')
    parser.add_argument('--no-show', action='store_true', help='do not open the web browser')
    parser.add_argument('--no-reproject', action='store_true',
                        help='display in the projection of the images, without basemap')
    parser.add_argument('--basemap', help='initial basemap, e.g. "CARTO Positron"')
    parser.add_argument('--cmap', help="initial colormap, e.g. 'cmo.turbid'")
    parser.add_argument('--stadia-api-key', help='API key of the Stadia Maps basemaps (or STADIA_API_KEY)')
    parser.add_argument('--cache-size', type=float, default=2.,
                        help='memory (GB) of the data kept in cache by each session: windows of the map with a '
                             'margin, coarser levels, blocks of the spectra and time series (default: 2)')
    parser.add_argument('--show-invalid-l2a', action='store_true',
                        help='show all the pixels of the L2A products (default: only the valid water pixels, '
                             'mask == 0)')
    parser.add_argument('--show-invalid-l2b', action='store_true',
                        help='show all the pixels of the L2B products (default: only the valid water pixels, '
                             'mask == 0)')
    series = parser.add_argument_group('time series of folders')
    series.add_argument('--tile', help='tile to open, e.g. 31TEJ (needed when the folders hold several UTM zones)')
    series.add_argument('--start', help='first acquisition date, e.g. 2024-01-01')
    series.add_argument('--end', help='last acquisition date (included), e.g. 2024-12-31')
    series.add_argument('--all-processings', action='store_true',
                        help='keep all the processings of each acquisition (default: the last one only)')
    parser.add_argument('--search-area', default=','.join(map(str, SEARCH_AREA)),
                        help='area shown by the search map at start-up: lon_min,lat_min,lon_max,lat_max, or "world" '
                             '(default: the Etang de Berre, %(default)s)')
    parser.add_argument('--stac', help='catalogue of the STAC search card: STAC API URL, or folder / glob pattern '
                                       'of STAC Items (e.g. the L2B products of GRSl2bgen --stac)')
    series.add_argument('--max-cloud-cover', type=float,
                        help='maximum cloud cover (%%) of the products with a STAC Item (eo:cloud_cover); the '
                             'products without Item are kept')
    args = parser.parse_args(argv)

    if args.stadia_api_key:
        from .visual import set_stadia_api_key
        set_stadia_api_key(args.stadia_api_key)
    serve(path=args.path, root=args.root, port=args.port, address=args.address, show=not args.no_show,
          allow_websocket_origin=args.allow_websocket_origin, reproject=not args.no_reproject,
          basemap=args.basemap, cmap=args.cmap, tile=args.tile, start=args.start, end=args.end,
          last_processing=not args.all_processings, mask_l2a=not args.show_invalid_l2a,
          mask_l2b=not args.show_invalid_l2b, cache_size=args.cache_size * 1e9,
          max_cloud_cover=args.max_cloud_cover, stac=args.stac,
          search_area=None if args.search_area == 'world' else tuple(map(float, args.search_area.split(','))))


if __name__ == '__main__':
    main()
