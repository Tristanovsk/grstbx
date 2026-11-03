"""
Static STAC catalogue of GRS L2A and GRSl2bgen L2B products, to publish them in an S3 bucket.

:func:`build_catalog` writes the catalogue, read by :func:`grstbx.app.search_items`::

    <out_dir>/catalog.json
    <out_dir>/grs-l2a/collection.json
    <out_dir>/grs-l2a/items.json                  all the Items of the collection, read at once by a search
    <out_dir>/grs-l2a/<id>/<id>.json
    <out_dir>/grs-l2b/...

The links between the files of the catalogue are relative, so that it can be moved; the assets are absolute
URLs in the bucket. The L2B Items are those written by GRSl2bgen ``--stac``; the Items of the products
without Item (L2A products, L2B NetCDF files written without ``--stac``) come from the attributes of the
products, with the same content. An L2B Item has a ``derived_from`` link to the Item of its L2A product.

The products can also be described from the bucket itself, when they were written there by a batch
(:class:`grstbx.batch.Batch` with ``stac=True`` and ``l2a_stac=True``): only their Items are read, and the
catalogue can be written to the bucket too.

Example
-------
>>> from grstbx import stac
>>> stac.build_catalog({'/data/L2A_zarr': 's3://bucket/L2A_zarr', '/data/L2B_zarr': 's3://bucket/L2B_zarr'},
...                    '/data/stac')
>>> stac.build_catalog(['s3://bucket/L2A_zarr', 's3://bucket/L2B_zarr'], 's3://bucket/stac')
"""

import datetime
import json
import os
import posixpath
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from .app import _read_item, find_products
from .batch import _fs, _write_text, is_url, l2a_complete
from .driver import is_zarr, open_zarr_image

STAC_VERSION = '1.0.0'
STAC_EXTENSIONS = [
    'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
    'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
    'https://stac-extensions.github.io/file/v2.1.0/schema.json',
    'https://stac-extensions.github.io/eo/v1.1.0/schema.json',
    'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
]
COLLECTIONS = {
    'L2A': {'id': 'grs-l2a', 'title': 'GRS L2A',
            'description': 'Remote sensing reflectance corrected from the atmosphere and the sunglint by GRS'},
    'L2B': {'id': 'grs-l2b', 'title': 'GRSl2bgen L2B',
            'description': 'Water quality parameters retrieved by GRSl2bgen from the GRS L2A products'},
}
MEDIA_JSON = 'application/json'
MEDIA_GEOJSON = 'application/geo+json'
MEDIA_TYPES = {'.nc': 'application/x-netcdf', '.zarr': 'application/vnd+zarr', '.log': 'text/plain'}


def _footprint(ds):
    """Polygon (lon/lat, counter-clockwise) of the grid of a product, and its bounding box."""
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = ds.rio.bounds()
    lon, lat = Transformer.from_crs(ds.rio.crs, 'EPSG:4326', always_xy=True).transform(
        [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])
    ring = [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)]
    bbox = [min(lon), min(lat), max(lon), max(lat)]
    return {'type': 'Polygon', 'coordinates': [ring + ring[:1]]}, [round(float(v), 7) for v in bbox]


def _isotime(value):
    """ISO date in UTC of an attribute; the dates without offset (``processing_time``) are in local time."""
    time = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return time.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


def product_item(path, level):
    """
    STAC Item of a product from its attributes, with the content of the Items of GRSl2bgen. The grid is
    that of the tile, whatever the part of it covered by the image. The assets hold local paths.

    :param path: Zarr store or NetCDF file
    :param level: ``'L2A'`` or ``'L2B'``
    :return: Item (dict)
    """
    ds = open_zarr_image(path) if is_zarr(path) else xr.open_dataset(path, decode_coords='all')
    with ds:
        attrs, crs, transform = ds.attrs, ds.rio.crs, ds.rio.transform()
        geometry, bbox = _footprint(ds)
        properties = {
            'constellation': str(attrs.get('constellation', '')).lower() or None,
            'platform': str(attrs.get('DATATAKE_1_SPACECRAFT_NAME', attrs.get('satellite', ''))).lower() or None,
            'instruments': [str(attrs['instrument']).lower()] if 'instrument' in attrs else None,
            'proj:code': crs.to_string(),
            'proj:shape': [int(ds.rio.height), int(ds.rio.width)],
            'proj:transform': [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            'proj:bbox': list(ds.rio.bounds()),
            'processing:level': level,
            'processing:software': {'GRS': str(attrs['version'])} if 'version' in attrs else None,
            'processing:datetime': _isotime(attrs['processing_time']) if 'processing_time' in attrs else None,
            'grid:code': f"MGRS-{attrs['tile']}" if 'tile' in attrs else None,
            'eo:cloud_cover': (float(attrs['CLOUD_COVERAGE_ASSESSMENT'])
                               if 'CLOUD_COVERAGE_ASSESSMENT' in attrs else None),
            'datetime': _isotime(attrs['acquisition_date']),
        }
    name, ext = os.path.splitext(os.path.basename(path))
    data = {'href': path, 'type': MEDIA_TYPES.get(ext), 'title': f"{COLLECTIONS[level]['title']} product",
            'roles': ['data']}
    assets = {'data': {**data, 'file:size': os.path.getsize(path)} if os.path.isfile(path) else data}
    log = os.path.join(os.path.dirname(path), 'log_file.log')
    if os.path.isfile(log):
        assets['log'] = {'href': log, 'type': 'text/plain', 'title': 'Processing log',
                         'file:size': os.path.getsize(log), 'roles': ['metadata']}
    return {'type': 'Feature', 'stac_version': STAC_VERSION, 'id': name,
            'properties': {k: v for k, v in properties.items() if v is not None},
            'geometry': geometry, 'bbox': bbox, 'links': [], 'assets': assets,
            'stac_extensions': list(STAC_EXTENSIONS)}


def _bucket_url(path, roots):
    """URL in the bucket of a local path below one of the `roots` (dict {folder: URL})."""
    path = os.path.abspath(path)
    for root, url in roots.items():
        root = os.path.abspath(root)
        if path == root or path.startswith(root + os.sep):
            return url.rstrip('/') + path[len(root):]
    raise ValueError(f'{path} is not below {list(roots)}')


def _catalog_item(product, roots):
    """
    Item of a row of :func:`~grstbx.app.find_products`, asset hrefs in the bucket, links not set yet.

    :return: (level, Item); the Item is None for an incomplete product
    """
    level, path, item_file = product.level, product.path, product.item
    if isinstance(item_file, str):
        item = _read_item(item_file)
        folder = os.path.dirname(item_file)
        for asset in item['assets'].values():
            if '://' not in asset['href']:
                asset['href'] = os.path.normpath(os.path.join(folder, asset['href']))
    elif level == 'L2B' and is_zarr(path):
        return level, None          # GRSl2bgen writes the Item last, after the product
    elif level == 'L2A' and not l2a_complete(path):
        return level, None
    else:
        item = product_item(path, level)
    for asset in item['assets'].values():
        if '://' not in asset['href']:
            asset['href'] = _bucket_url(asset['href'], roots)
    return level, item


def _resolve(folder, href):
    """Absolute URL of a relative href of a file of the folder `folder` (URL)."""
    protocol, path = folder.split('://', 1)
    return f"{protocol}://{posixpath.normpath(posixpath.join(path, href))}"


def bucket_items(root, storage_options=None):
    """
    STAC Items of the products of a bucket folder: ``<root>/<tile>/YYYY/MM/DD/<name>/<name>.json`` (product
    folders of the batches) or ``<root>/<tile>/YYYY/MM/DD/<name>.json``, with their asset hrefs made absolute
    URLs. The products without Item are not seen.

    :param root: URL of the folder, e.g. ``s3://bucket/L2B_zarr``
    :param storage_options: options of the fsspec file system (default: from the environment)
    :return: list of (level, Item), level ``'L2A'`` or ``'L2B'``
    """
    fs, path = _fs(root, storage_options)
    entries = fs.glob(path.rstrip('/') + '/*/*/*/*/*')
    candidates = [p for p in entries if p.endswith('.json')] + \
                 [f'{p}/{posixpath.basename(p)}.json' for p in entries
                  if not p.endswith(('.json', '.zarr', '.nc', '.log', '.csv'))]
    contents = fs.cat(candidates, on_error='omit') if candidates else {}
    items = []
    for file in sorted(contents):
        try:
            item = json.loads(contents[file])
        except ValueError:
            continue
        if not (isinstance(item, dict) and item.get('type') == 'Feature' and 'stac_version' in item):
            continue
        level = item.get('properties', {}).get('processing:level') or ('L2B' if '_MSIL2B_' in item['id'] else 'L2A')
        folder = posixpath.dirname(fs.unstrip_protocol(file))
        for asset in item.get('assets', {}).values():
            if '://' not in asset['href']:
                asset['href'] = _resolve(folder, asset['href'])
        items.append((level, item))
    return items


def _derived_from(item, l2a_ids):
    """Id of the Item of the L2A product of an L2B Item, None if it is not in the catalogue."""
    l2a_id = item['id'].replace('_MSIL2B_', '_MSIL2AGRS_')
    return l2a_id if l2a_id in l2a_ids else None


def _extent(items):
    """Extent of a collection: bounding box and period of its Items."""
    boxes = np.array([item['bbox'] for item in items])
    times = sorted(item['properties']['datetime'] for item in items)
    return {'spatial': {'bbox': [[*boxes[:, :2].min(axis=0).tolist(), *boxes[:, 2:].max(axis=0).tolist()]]},
            'temporal': {'interval': [[times[0], times[-1]]]}}


def _write_json(path, content, storage_options=None):
    if is_url(path):
        _write_text(path, json.dumps(content, indent=1), storage_options)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.part{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(content, f, indent=1)
    os.replace(tmp, path)


def build_catalog(roots, out_dir, title='GRS products', description='GRS L2A and GRSl2bgen L2B products',
                  license='proprietary', workers=8, storage_options=None):
    """
    Write the static STAC catalogue of the products (see :mod:`grstbx.stac`), described with their URLs in
    the bucket. Run it again when products are added: the catalogue is rewritten from the products; the Item
    files of the products removed stay but are no more linked.

    :param roots: dict {local folder of products: URL of this folder in the bucket}, e.g.
                  ``{'/data/L2A_zarr': 's3://bucket/L2A_zarr'}``; or URLs of folders of the bucket (list, or
                  dict keys), whose products are described from their Items alone (:func:`bucket_items`)
    :param out_dir: folder of the catalogue: local, to publish in the bucket, or URL in the bucket
    :param title: title of the catalogue
    :param description: description of the catalogue
    :param license: license of the collections (SPDX identifier, ``proprietary`` or ``various``)
    :param workers: number of local products read in parallel
    :param storage_options: options of the fsspec file system of the URLs (default: from the environment)
    :return: (Items per collection, list of the local products skipped as incomplete)
    """
    if not isinstance(roots, dict):
        roots = {root: root for root in ([roots] if isinstance(roots, str) else roots)}
    local = {root: url for root, url in roots.items() if not is_url(root)}
    results = [result for root in roots if is_url(root) for result in bucket_items(root, storage_options)]
    skipped = []
    if local:
        products = pd.concat([find_products(root) for root in local], ignore_index=True)
        products = products[products.level.isin(COLLECTIONS)]
        with ThreadPoolExecutor(workers) as pool:
            found = list(pool.map(lambda row: _catalog_item(row, local), products.itertuples()))
        skipped = [path for path, (_, item) in zip(products.path, found) if item is None]
        results += found
    items = {level: sorted((item for lev, item in results if lev == level and item is not None),
                           key=lambda item: item['properties']['datetime']) for level in COLLECTIONS}
    l2a_ids = {item['id'] for item in items['L2A']}

    catalog_links = [{'rel': 'root', 'href': '../catalog.json', 'type': MEDIA_JSON},
                     {'rel': 'parent', 'href': '../catalog.json', 'type': MEDIA_JSON}]
    children = []
    for level, level_items in items.items():
        if not level_items:
            continue
        spec = COLLECTIONS[level]
        collection_id = spec['id']
        folder = os.path.join(out_dir, collection_id)
        for item in level_items:
            item['collection'] = collection_id
            item['links'] = [{'rel': 'root', 'href': '../../catalog.json', 'type': MEDIA_JSON},
                             {'rel': 'parent', 'href': '../collection.json', 'type': MEDIA_JSON},
                             {'rel': 'collection', 'href': '../collection.json', 'type': MEDIA_JSON}]
            l2a_id = _derived_from(item, l2a_ids) if level == 'L2B' else None
            if l2a_id:
                item['links'].append({'rel': 'derived_from', 'type': MEDIA_GEOJSON, 'title': 'GRS L2A product',
                                      'href': f"../../{COLLECTIONS['L2A']['id']}/{l2a_id}/{l2a_id}.json"})
            _write_json(os.path.join(folder, item['id'], item['id'] + '.json'), item, storage_options)
        # in items.json too, the links are relative to the folder of each Item: <id>/
        _write_json(os.path.join(folder, 'items.json'), {'type': 'FeatureCollection', 'features': level_items},
                    storage_options)
        _write_json(os.path.join(folder, 'collection.json'), storage_options=storage_options, content={
            'type': 'Collection', 'stac_version': STAC_VERSION, 'id': collection_id, 'title': spec['title'],
            'description': spec['description'], 'license': license, 'extent': _extent(level_items),
            'summaries': {
                'processing:level': [level],
                'grid:code': sorted({item['properties'].get('grid:code') for item in level_items} - {None}),
                'platform': sorted({item['properties'].get('platform') for item in level_items} - {None})},
            'links': catalog_links
            + [{'rel': 'items', 'href': './items.json', 'type': MEDIA_GEOJSON}]
            + [{'rel': 'item', 'href': f"./{item['id']}/{item['id']}.json", 'type': MEDIA_GEOJSON}
               for item in level_items]})
        children.append({'rel': 'child', 'href': f'./{collection_id}/collection.json', 'type': MEDIA_JSON,
                         'title': spec['title']})
    _write_json(os.path.join(out_dir, 'catalog.json'), storage_options=storage_options, content={
        'type': 'Catalog', 'stac_version': STAC_VERSION, 'id': 'grs-products', 'title': title,
        'description': description,
        'links': [{'rel': 'root', 'href': './catalog.json', 'type': MEDIA_JSON}] + children})
    return {COLLECTIONS[level]['id']: len(level_items) for level, level_items in items.items()}, skipped
