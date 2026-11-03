"""
Search and download of Sentinel-2 images from the `Copernicus Data Space Ecosystem
<https://dataspace.copernicus.eu>`_ (CDSE), into the tree of the batches: ``<odir>/<tile>/YYYY/MM/DD/<image>.SAFE``
(see :func:`grstbx.batch.list_images`).

The search uses the public OData catalogue of CDSE (no account needed). The images are downloaded from
the S3 interface of CDSE (``eodata`` bucket, https://documentation.dataspace.copernicus.eu/APIs/S3.html),
which needs S3 keys of a CDSE account: from ``storage_options``, an AWS profile or the ``~/.s3cfg`` file
of s3cmd (see :func:`cdse_storage_options`). The download needs s3fs (``pip install "grstbx[remote]"``).

.. code-block:: python

    from grstbx import download

    images = download.search(['31TFJ'], '2024-07-01', '2024-07-31', max_cloud_cover=20)
    summary = download.download(images, '/data/satellite/Sentinel-2/L1C')
"""

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

#: OData catalogue of the Copernicus Data Space Ecosystem
CATALOGUE = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
#: S3 endpoint of the Copernicus Data Space Ecosystem (bucket ``eodata``)
S3_ENDPOINT = 'https://eodata.dataspace.copernicus.eu'
#: Sentinel-2 product types of the catalogue
PRODUCT_TYPES = {'L1C': 'S2MSI1C', 'L2A': 'S2MSI2A'}

# Sentinel-2 names: <mission>_MSI<level>_<sensing time>_N<baseline>_R<orbit>_T<tile>_<processing time>.SAFE
_NAME = re.compile(r'^(S2[A-D])_MSI(\w+?)_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})')
COLUMNS = ['name', 'tile', 'time', 'satellite', 'baseline', 'processing_time', 'cloud_cover', 'size', 'online',
           's3_path', 'id', 'footprint']


def _read_json(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _isotime(value):
    return pd.Timestamp(value).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def _wkt(area):
    """WKT polygon (lon/lat) of an area: shapely geometry or (lon_min, lat_min, lon_max, lat_max)."""
    from shapely.geometry import box

    geometry = box(*area) if isinstance(area, (list, tuple)) else area
    return geometry.wkt


def _filter(start, end, tile=None, area=None, product_type='S2MSI1C', max_cloud_cover=None):
    """``$filter`` of an OData search of the CDSE catalogue."""
    terms = ["Collection/Name eq 'SENTINEL-2'",
             "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
             f"att/OData.CSC.StringAttribute/Value eq '{product_type}')",
             f'ContentDate/Start ge {_isotime(start)}',
             f"ContentDate/Start lt {_isotime(pd.Timestamp(end) + pd.Timedelta('1D'))}"]
    if tile:
        terms.append(f"contains(Name,'_T{tile}_')")
    if area is not None:
        terms.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{_wkt(area)}')")
    if max_cloud_cover is not None:
        terms.append("Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
                     f"att/OData.CSC.DoubleAttribute/Value le {float(max_cloud_cover):.2f})")
    return ' and '.join(terms)


def _row(product):
    """Row of the table of :func:`search` from a product of the catalogue."""
    attributes = {attribute['Name']: attribute['Value'] for attribute in product.get('Attributes', [])}
    match = _NAME.match(product['Name'])
    mission, _, sensing, baseline, _, tile, processing = match.groups() if match else (None,) * 7
    return {'name': product['Name'],
            'tile': attributes.get('tileId', tile),
            'time': pd.Timestamp(product['ContentDate']['Start']).tz_localize(None),
            'satellite': mission,
            'baseline': baseline,
            'processing_time': pd.Timestamp(processing) if processing else pd.NaT,
            'cloud_cover': attributes.get('cloudCover', np.nan),
            'size': product.get('ContentLength'),
            'online': product.get('Online'),
            's3_path': product.get('S3Path'),
            'id': product.get('Id'),
            'footprint': product.get('GeoFootprint')}


def search(tiles=None, start=None, end=None, area=None, max_cloud_cover=None, level='L1C', last_processing=True,
           top=1000):
    """
    Sentinel-2 images of the CDSE catalogue, by tile or by area, between two dates.

    :param tiles: tile or list of tiles (e.g. ``['31TFJ', '31TGJ']``); None: all the tiles of ``area``
    :param start: first acquisition date (included)
    :param end: last acquisition date (included)
    :param area: area (shapely geometry or (lon_min, lat_min, lon_max, lat_max), lon/lat): images whose
                 footprint intersects it
    :param max_cloud_cover: maximum cloud cover of the image (%)
    :param level: ``'L1C'`` or ``'L2A'`` (or a product type of the catalogue, e.g. ``'S2MSI1C'``)
    :param last_processing: keep the last processing of each image (the reprocessings of the archive, e.g.
                            baseline 05.00, give several products for the same acquisition and tile)
    :param top: number of products per request (the following pages are read too)
    :return: pandas.DataFrame, one row per image, sorted by acquisition time: name, tile, time, satellite,
             baseline, processing_time, cloud_cover (%), size (bytes), online, s3_path, id, footprint
    """
    if start is None or end is None:
        raise ValueError('give the period: start and end')
    if tiles is None and area is None:
        raise ValueError('give tiles or an area')
    product_type = PRODUCT_TYPES.get(level, level)
    rows = []
    for tile in ([tiles] if isinstance(tiles, str) else list(tiles)) if tiles is not None else [None]:
        query = {'$filter': _filter(start, end, tile, area, product_type, max_cloud_cover),
                 '$orderby': 'ContentDate/Start', '$top': top, '$expand': 'Attributes'}
        url = f'{CATALOGUE}?{urllib.parse.urlencode(query, quote_via=urllib.parse.quote)}'
        while url:
            page = _read_json(url)
            rows += [_row(product) for product in page.get('value', [])]
            url = page.get('@odata.nextLink')
    images = pd.DataFrame(rows, columns=COLUMNS)
    if last_processing and len(images):
        images = (images.sort_values('processing_time', kind='stable')
                  .drop_duplicates(['satellite', 'time', 'tile'], keep='last'))
    return images.drop_duplicates('name').sort_values(['time', 'tile'], kind='stable').reset_index(drop=True)


def level1_dir(odir, name):
    """
    Folder of an image in the tree of the batches: ``<odir>/<tile>/YYYY/MM/DD`` (acquisition date of the
    name), as read by :func:`grstbx.batch.list_images`.
    """
    match = _NAME.match(os.path.basename(str(name)))
    if not match:
        raise ValueError(f'{name} is not a Sentinel-2 product name')
    sensing, tile = match.group(3), match.group(6)
    return os.path.join(odir, tile, sensing[:4], sensing[4:6], sensing[6:8])


def _s3cfg(path):
    """Keys of an s3cmd configuration file."""
    keys = {}
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if '=' in line:
                key, value = (part.strip() for part in line.split('=', 1))
                keys[key] = value
    return keys


def cdse_storage_options(profile=None, s3cfg='~/.s3cfg'):
    """
    Options of the s3fs file system of the CDSE S3 interface: keys of an AWS profile (``profile``), or else
    of the s3cmd configuration file ``s3cfg`` when it exists (``access_key``, ``secret_key``), or else of the
    environment (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``). The keys are created in the CDSE account
    (https://eodata-s3keysmanager.dataspace.copernicus.eu).

    :return: dict for :func:`download` (``storage_options``)
    """
    options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT}}
    if profile:
        options['profile'] = profile
    elif s3cfg and os.path.isfile(os.path.expanduser(s3cfg)):
        keys = _s3cfg(s3cfg)
        if keys.get('access_key') and keys.get('secret_key'):
            options.update(key=keys['access_key'], secret=keys['secret_key'])
    return options


def s3cmd_commands(images, odir, config='~/.s3cfg'):
    """
    ``s3cmd sync`` commands downloading the images into the tree of the batches, e.g. to run them from a
    shell or a scheduler instead of :func:`download`.

    :param images: table of :func:`search`
    :return: list of commands
    """
    return [f's3cmd -c {config} sync s3:/{row.s3_path} {level1_dir(odir, row.name)}/' for row in images.itertuples()]


def _complete(path):
    """True if an image folder holds its metadata (written last by :func:`download`)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, 'manifest.safe'))


def on_disk(images, odir):
    """
    Images of a search already downloaded into the tree of the batches (folder with its
    ``manifest.safe``).

    :param images: table of :func:`search`
    :return: pandas.Series of booleans, indexed as ``images``
    """
    return pd.Series([_complete(os.path.join(level1_dir(odir, name), name)) for name in images.name],
                     index=images.index, dtype=bool)


def download(images, odir, storage_options=None, workers=4, overwrite=False, fs=None):
    """
    Download images from the CDSE S3 interface into ``<odir>/<tile>/YYYY/MM/DD/<image>.SAFE``.

    Each image is written to ``<image>.SAFE.part`` and renamed at the end, so that an interrupted download
    leaves no incomplete image under its name: the images already downloaded are skipped, the others are
    downloaded again at the next call.

    :param images: table of :func:`search` (columns name and s3_path)
    :param odir: root folder of the Level-1 images
    :param storage_options: options of the s3fs file system (default: :func:`cdse_storage_options`)
    :param workers: images downloaded at the same time
    :param overwrite: download the images already on disk again
    :param fs: fsspec file system of the images (default: s3fs on the CDSE endpoint)
    :return: pandas.DataFrame, one row per image: name, path, status (``downloaded``, ``exists``, ``failed``),
             error
    """
    if fs is None:
        import s3fs

        fs = s3fs.S3FileSystem(**(storage_options or cdse_storage_options()))

    def one(row):
        path = os.path.join(level1_dir(odir, row.name), row.name)
        result = {'name': row.name, 'path': path, 'status': 'exists', 'error': ''}
        if _complete(path) and not overwrite:
            return result
        part = path + '.part'
        try:
            shutil.rmtree(part, ignore_errors=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fs.get(row.s3_path.lstrip('/').rstrip('/') + '/', part + '/', recursive=True)
            if not _complete(part):
                raise OSError('manifest.safe missing after the download')
            shutil.rmtree(path, ignore_errors=True)
            os.rename(part, path)
            result['status'] = 'downloaded'
        except Exception as exc:     # a failing image does not stop the others
            shutil.rmtree(part, ignore_errors=True)
            result.update(status='failed', error=str(exc))
        return result

    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = list(pool.map(one, images.itertuples()))
    return pd.DataFrame(rows, columns=['name', 'path', 'status', 'error'])
