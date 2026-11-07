"""
Batch processing of Level-1 images into GRS L2A products and GRSl2bgen L2B products, image by image.

The functions list the inputs found in the usual ``<root>/<tile>/YYYY/MM/DD/<product>`` trees, name the
outputs as the ``grs`` and ``GRSl2bgen`` command lines do, check that a product is complete, and profile
the resources of each processing step. :class:`Batch` combines them into the three batches of the
tutorials:

- Level-1 (Sentinel-2 L1C, Landsat Collection 2) to GRS L2A, with GRS 2.x (NetCDF products) or GRS 3.x
  (Zarr products): :meth:`Batch.plan_level1` with ``l2b_dir=None``;
- Level-1 to L2A then L2B, image by image: :meth:`Batch.plan_level1` with an ``l2b_dir``;
- GRS L2A to GRSl2bgen L2B: :meth:`Batch.plan_l2a` on the products of :func:`list_l2a_products`.

Example
-------
>>> from grstbx import batch
>>> images = batch.list_images('/data/L1C', ['31TEJ']).loc['2025-08-01':'2025-08-04']
>>> run = batch.Batch('/data/L2A', '/data/L2B', cams_file_template='/data/cams/cams_{date:%Y-%m}.nc')
>>> plan = run.plan_level1(images)        # state of the products and steps to run, one row per image
>>> summary = run.run(plan)               # GRS then GRSl2bgen, a profile per image

The ``grs`` and ``GRSl2bgen`` packages are imported only to process the images.

Buckets
-------
The roots of the inputs and of the outputs can be URLs of a bucket (``s3://bucket/L1C``,
``s3://bucket/L2A_zarr``..., or any fsspec file system), read and written through fsspec with the
credentials and endpoint of the environment (e.g. ``AWS_PROFILE``) or the ``storage_options`` given. GRS
and GRSl2bgen work on local files: :meth:`Batch.process` downloads the inputs of each image to a local
work folder, processes them there, uploads the product folders (the STAC Item of an L2B product last, so
that it marks a complete product in the bucket too) and removes the local copies.

>>> images = batch.list_images('s3://bucket/L1C', ['31TEJ'])
>>> run = batch.Batch('s3://bucket/L2A_zarr', 's3://bucket/L2B_zarr', work_dir='/scratch/grstbx',
...                   cams_file_template='/data/cams/cams_{date:%Y-%m}.nc', l2a_stac=True)
"""

import os
import re
import gc
import glob
import json
import time
import shutil
import socket
import logging
import platform
import tempfile
import threading
import contextlib
import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

opj = os.path.join


# ---------------------------------------------------------------------------------------------------
# local folders and buckets
# ---------------------------------------------------------------------------------------------------

def is_url(path):
    """True if `path` is a URL (``s3://``, ``gs://``, ``memory://``...), read and written through fsspec."""
    return '://' in str(path)


def _fs(path, storage_options=None):
    """fsspec file system of a URL and the path in it."""
    import fsspec

    return fsspec.core.url_to_fs(str(path), **(storage_options or {}))


def _glob(pattern, storage_options=None):
    """Paths (URLs for a bucket) matching a glob pattern, sorted."""
    if not is_url(pattern):
        return sorted(glob.glob(str(pattern)))
    fs, path = _fs(pattern, storage_options)
    return sorted(fs.unstrip_protocol(p) for p in fs.glob(path))


def exists(path, storage_options=None):
    """True if a local path or URL exists."""
    if not is_url(path):
        return os.path.exists(path)
    fs, path = _fs(path, storage_options)
    return fs.exists(path)


def _isdir(path, storage_options=None):
    if not is_url(path):
        return os.path.isdir(path)
    fs, path = _fs(path, storage_options)
    return fs.isdir(path)


def _listdir(path, storage_options=None):
    """Names of the entries of a folder."""
    if not is_url(path):
        return os.listdir(path)
    fs, path = _fs(path, storage_options)
    return [os.path.basename(p.rstrip('/')) for p in fs.ls(path, detail=False)]


def _read_text(path, storage_options=None):
    if not is_url(path):
        with open(path) as f:
            return f.read()
    fs, path = _fs(path, storage_options)
    return fs.cat_file(path).decode()


def _write_text(path, text, storage_options=None):
    if not is_url(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)
        return
    fs, path = _fs(path, storage_options)
    fs.pipe_file(path, text.encode())


def download(url, local, storage_options=None):
    """
    Copy a file or folder of a bucket to the local path `local` (replaced if it exists).

    :return: `local`
    """
    remove_output(local)
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    fs, path = _fs(url, storage_options)
    if fs.isdir(path):
        fs.get(path.rstrip('/') + '/', local, recursive=True)
    else:
        fs.get(path, local)
    return local


def upload(local, url, storage_options=None, last=None):
    """
    Copy a local file or folder to the bucket at `url` (replaced if it exists). The Zarr metadata files
    are uploaded after the chunks, the deepest first, so that a store interrupted during the upload is
    not complete (:func:`zarr_complete`).

    :param last: file of the folder uploaded last, once all the others are (e.g. the STAC Item marking a
                 complete product)
    """
    remove_output(url, storage_options)
    fs, path = _fs(url, storage_options)
    if not os.path.isdir(local):
        fs.put_file(local, path)
        return
    metadata = {'zarr.json', '.zmetadata', '.zgroup', '.zarray', '.zattrs'}
    last = os.path.abspath(last) if last is not None else None
    files = sorted((p for p in Path(local).rglob('*') if p.is_file()),
                   key=lambda p: (str(p.absolute()) == last, p.name in metadata, -len(p.parts), str(p)))
    files = [str(p) for p in files]
    for file in files:
        fs.put_file(file, path.rstrip('/') + '/' + Path(file).relative_to(local).as_posix())

#: fields of the names of the GRS L2A products (the version is optional)
FIELDS = ['satellite', 'level', 'time', 'baseline', 'orbit', 'tile', 'processing_time', 'version']
#: variables an L2A product must hold to be complete
REQUIRED = ['Rrs', 'mask']


# ---------------------------------------------------------------------------------------------------
# Level-1 images
# ---------------------------------------------------------------------------------------------------

def cloud_cover(path, storage_options=None):
    """
    Cloud cover of a Level-1 image, from its metadata (``MTD_MSIL1C.xml`` of Sentinel-2, ``*_MTL.txt``
    of Landsat).

    :param path: Level-1 image folder (local or URL)
    :param storage_options: options of the fsspec file system of a URL
    :return: cloud cover (0-1), NaN if not found
    """
    patterns = [('MTD_MSIL1C.xml', r'<Cloud_Coverage_Assessment>\s*([\d.]+)'),
                ('*_MTL.txt', r'\bCLOUD_COVER\s*=\s*"?([\d.]+)')]
    for metadata, regex in patterns:
        for file in _glob(opj(path, metadata), storage_options):
            match = re.search(regex, _read_text(file, storage_options))
            if match:
                return float(match.group(1)) / 100
    return np.nan


def list_images(idir, tiles, pattern='S2*.SAFE', storage_options=None):
    """
    Table of the Level-1 images found in ``<idir>/<tile>/YYYY/MM/DD/``.

    :param idir: root folder of the Level-1 images, local or URL of a bucket (``s3://bucket/L1C``)
    :param tiles: tile sub-folders (e.g. ``['31TEJ']``, or Landsat path/rows ``['199027']``)
    :param pattern: glob pattern of the image names (``'L*_L1*'`` for Landsat)
    :param storage_options: options of the fsspec file system of a URL (default: from the environment)
    :return: pandas.DataFrame indexed by acquisition date (from the folders), with the columns
             ``tile``, ``name``, ``path`` and ``cloud_cover`` (0-1)
    """
    rows = []
    for tile in tiles:
        for path in _glob(opj(idir, tile, '*', '*', '*', pattern), storage_options):
            year, month, day = Path(path).parts[-4:-1]
            rows.append(dict(date=pd.Timestamp(f'{year}-{month}-{day}'), tile=tile,
                             name=os.path.basename(path.rstrip('/')), path=path))
    df = pd.DataFrame(rows, columns=['date', 'tile', 'name', 'path'])
    df['cloud_cover'] = df['path'].map(partial(cloud_cover, storage_options=storage_options)).astype(float)
    return df.set_index('date').sort_index()


# ---------------------------------------------------------------------------------------------------
# names of the products
# ---------------------------------------------------------------------------------------------------

def l2a_name(level1_name, levname='L2AGRS', suffix=''):
    """
    Name of the L2A product of a Level-1 image, as with the ``grs`` command line (``set_ofile``): level
    of the image (``L1C``, ``L1GT``, ``L1TP``) replaced by `levname`, extension removed, plus `suffix`.

    >>> l2a_name('S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE', suffix='_V3.1.0')
    'S2A_MSIL2AGRS_20250801T104041_N0511_R008_T31TEJ_20250801T131318_V3.1.0'
    """
    name = os.path.basename(str(level1_name).rstrip('/'))
    for level in ('L1C', 'L1GT', 'L1TP'):
        name = name.replace(level, levname)
    return Path(name).stem + suffix


def l2b_name(l2a):
    """
    Name of the L2B product of an L2A product, as with the ``GRSl2bgen`` command line
    (``default_output_name``): extension removed and level tag ``L2AGRS`` or ``L2A`` (any case)
    replaced by ``L2B``.

    >>> l2b_name('/data/S2A_MSIL2AGRS_20250801T104041_N0511_R008_T31TEJ_20250801T131318_V3.1.0.zarr')
    'S2A_MSIL2B_20250801T104041_N0511_R008_T31TEJ_20250801T131318_V3.1.0'
    """
    name = re.sub(r'\.(nc|zarr)$', '', os.path.basename(str(l2a).rstrip('/')))
    return re.sub('L2A(grs)?', 'L2B', name, count=1, flags=re.IGNORECASE)


def product_name(path):
    """Name of a product: file, store or folder name without the ``.nc`` / ``.zarr`` extension."""
    name = os.path.basename(str(path).rstrip('/'))
    for ext in ('.nc', '.zarr'):
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name


def image_name(path):
    """Name of an input image: Level-1 (``.SAFE`` removed) or L2A product (``.nc`` / ``.zarr``
    removed); the dots of a version (``_V3.1.0``) are kept."""
    return re.sub(r'\.(SAFE|nc|zarr)$', '', os.path.basename(str(path).rstrip('/')))


def l2a_format(path, storage_options=None):
    """
    Format of an L2A product: ``'zarr'`` or ``'nc'``, None if unknown.

    :param path: product file or store, or GRS product folder ``<name>/`` holding ``<name>.zarr`` or
                 ``<name>.nc`` (local or URL)
    """
    path = str(path).rstrip('/')
    for ext in ('zarr', 'nc'):
        if path.endswith('.' + ext) or exists(opj(path, f'{product_name(path)}.{ext}'), storage_options):
            return ext
    return None


def zarr_store(path):
    """Zarr store of a product: `path` itself if it ends with ``.zarr``, else ``<path>/<name>.zarr``."""
    path = str(path).rstrip('/')
    return path if path.endswith('.zarr') else opj(path, product_name(path) + '.zarr')


def item_path(product):
    """STAC Item of a product, next to it: ``<name>.json`` (written by GRSl2bgen for an L2B product, by
    :func:`write_l2a_item` for an L2A product)."""
    return os.path.splitext(str(product).rstrip('/\\'))[0] + '.json'


def dem_path(template, tile):
    """
    DEM file of a tile.

    :param template: path formatted with ``tile`` (e.g. ``'/data/dem/COP-DEM_{tile}.tif'``), or None
    :return: the path if the file exists, else None
    """
    if template is None:
        return None
    file = template.format(tile=tile)
    return file if os.path.exists(file) else None


# ---------------------------------------------------------------------------------------------------
# complete products
# ---------------------------------------------------------------------------------------------------

def zarr_complete(path, required=REQUIRED, storage_options=None):
    """
    True if a Zarr L2A product is complete: root metadata and, for each level of its pyramid (read from
    the ``multiscales`` layout of the root, or the numbered groups), the `required` arrays. A GRS run that
    failed while writing the store leaves the level groups without them.

    :param path: Zarr store, or GRS product folder holding it (local or URL)
    """
    store = zarr_store(path)
    try:
        attrs = json.loads(_read_text(opj(store, 'zarr.json'), storage_options)).get('attributes', {})
    except (OSError, ValueError):
        return False
    layout = (attrs.get('multiscales') or [{}])[0].get('layout') or []
    groups = [level['group'] for level in layout] or sorted(d for d in _listdir(store, storage_options)
                                                            if d.isdigit())
    return bool(groups) and all(exists(opj(store, group, *var, 'zarr.json'), storage_options)
                                for group in groups for var in [()] + [(v,) for v in required])


def _open_netcdf(path, storage_options=None):
    import xarray as xr

    if not is_url(path):
        return xr.open_dataset(path)
    fs, path = _fs(path, storage_options)
    return xr.open_dataset(fs.open(path), engine='h5netcdf')


def netcdf_complete(path, required=REQUIRED, storage_options=None):
    """
    True if a NetCDF L2A product is complete: ``<name>.nc`` opens and holds the `required` variables
    and, for a GRS product folder, its ancillary file ``<name>_anc.nc`` opens (a truncated file does not).

    :param path: ``<name>.nc`` file (its ancillary file is checked when it exists next to it), or GRS
                 product folder ``<name>/`` (local or URL; in a bucket, read with h5netcdf)
    """
    path = str(path).rstrip('/')
    if path.endswith('.nc'):
        main, ancillary = path, path[:-len('.nc')] + '_anc.nc'
        if not exists(ancillary, storage_options):
            ancillary = None
    else:
        name = product_name(path)
        main, ancillary = opj(path, name + '.nc'), opj(path, name + '_anc.nc')
    try:
        with _open_netcdf(main, storage_options) as ds:
            if not set(required) <= set(ds.variables):
                return False
        if ancillary is not None:
            with _open_netcdf(ancillary, storage_options):
                pass
    except Exception:
        return False
    return True


def l2a_complete(path, required=REQUIRED, storage_options=None):
    """True if the L2A product of `path` (Zarr or NetCDF, file, store or GRS folder, local or URL) is
    complete."""
    fmt = l2a_format(path, storage_options)
    if fmt == 'zarr':
        return zarr_complete(path, required, storage_options)
    if fmt == 'nc':
        return netcdf_complete(path, required, storage_options)
    return False


def l2b_complete(path, stac=True, storage_options=None):
    """
    True if an L2B product is complete: its STAC Item exists (GRSl2bgen writes it last, after the product
    and its log), or the product itself if `stac` is False (GRSl2bgen writes it under a temporary name
    and renames it when complete).
    """
    return exists(item_path(path) if stac else path, storage_options)


def l2b_to_zarr(path, ofile=None, stac=True, overwrite=False, chunks=512):
    """
    Convert a GRSl2bgen L2B NetCDF product into a multiscale (pyramid) Zarr store, as written by GRSl2bgen
    ``--format zarr`` (:meth:`GRSl2bgen.output.L2bProduct.export_to_zarr_pyramid`): one group per level,
    chunks of ``chunks`` pixels, consolidated metadata. The packing of the NetCDF variables (``int16`` with
    ``scale_factor`` and ``add_offset``, class indexes in ``uint8``) is kept, so that the values are the same.

    :param path: L2B NetCDF file
    :param ofile: Zarr store to write; default: in a product folder next to the file,
                  ``<dir>/<name>/<name>.zarr`` (layout of GRSl2bgen ``--product_dir``)
    :param stac: write the STAC Item of the store next to it, last, with its quick look
                 (:func:`GRSl2bgen.stac.write_stac_item`), so that the product is complete
                 (:func:`l2b_complete`)
    :param overwrite: convert again a product already complete
    :return: path of the store
    """
    import xarray as xr
    from GRSl2bgen.output import L2bProduct
    from GRSl2bgen.stac import write_stac_item

    name = os.path.splitext(os.path.basename(path))[0]
    ofile = ofile or opj(os.path.dirname(path), name, f'{name}.zarr')
    if l2b_complete(ofile, stac=stac) and not overwrite:
        return ofile
    # read by the chunks of the file (each decompressed once), rechunked by the writer
    ds = xr.open_dataset(path, decode_coords='all', chunks={})
    keep = ('dtype', 'scale_factor', 'add_offset', '_FillValue')
    encoding = {name_: {**{k: v for k, v in ds[name_].encoding.items() if k in keep and v is not None},
                        'grid_mapping': 'spatial_ref'} for name_ in ds.data_vars}
    # the writer of GRSl2bgen, with the product and the packing of the NetCDF file
    product = L2bProduct.__new__(L2bProduct)
    product.l2b_prod, product.variables = ds, list(ds.data_vars)
    product._packing_encoding = lambda persist=False: encoding
    os.makedirs(os.path.dirname(ofile), exist_ok=True)
    export = product.export_to_zarr_pyramid
    product.export_to_zarr_pyramid = lambda tmp, persist=False: export(tmp, persist=persist,
                                                                        chunks={'y': chunks, 'x': chunks})
    product.export(ofile, pyramid=True)
    ds.close()
    if stac:
        write_stac_item(ofile)
    return ofile


def remove_output(path, storage_options=None):
    """Remove a product: file, or folder (Zarr store, product folder), local or in a bucket; nothing if
    it does not exist."""
    if is_url(path):
        fs, path = _fs(path, storage_options)
        if fs.exists(path):
            fs.rm(path, recursive=True)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


# ---------------------------------------------------------------------------------------------------
# L2A products
# ---------------------------------------------------------------------------------------------------

def list_l2a_products(rootdir, tiles=None, pattern='*L2A*', check=True, storage_options=None):
    """
    Table of the GRS L2A products found in ``<rootdir>/<tile>/YYYY/MM/DD/``: product folders
    ``<name>/`` (holding ``<name>.zarr`` or ``<name>.nc``), or stores and files directly in the ``DD``
    folders.

    The names follow the Sentinel-2 convention followed by the GRS version,
    ``<satellite>_<level>_<sensing time>_<baseline>_<orbit>_<tile>_<processing time>_<version>``.

    :param rootdir: root folder of the L2A products, local or URL of a bucket
    :param tiles: tile sub-folders, None for all those of `rootdir`
    :param pattern: glob pattern of the product names
    :param check: add the column ``complete`` (:func:`l2a_complete`)
    :param storage_options: options of the fsspec file system of a URL (default: from the environment)
    :return: pandas.DataFrame indexed by sensing time, with the fields of the names, ``format``
             (``'zarr'`` or ``'nc'``), ``path`` (and ``complete``)
    """
    if tiles is None:
        tiles = sorted(d for d in _listdir(rootdir, storage_options)
                       if re.fullmatch(r'\d{2}[A-Z]{3}|\d{6}', d) and _isdir(opj(rootdir, d), storage_options))
    paths = sorted(p for tile in tiles
                   for p in _glob(opj(rootdir, tile, '*', '*', '*', pattern), storage_options))
    rows = []
    for path in paths:
        if product_name(path).endswith('_anc') or path.endswith('.json'):   # ancillary file, STAC Item
            continue
        fields = product_name(path).split('_')
        if len(fields) < 7:
            logging.warning(f'unexpected product name, skipped: {path}')
            continue
        fields = (fields + [None])[:len(FIELDS)]         # the version field is optional
        rows.append(dict(zip(FIELDS, fields), format=l2a_format(path, storage_options), path=path))
    df = pd.DataFrame(rows, columns=FIELDS + ['format', 'path'])
    df['tile'] = df['tile'].str.lstrip('T')
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%dT%H%M%S')
    if check:
        df['complete'] = df['path'].map(partial(l2a_complete, storage_options=storage_options)).astype(bool)
    return df.set_index('time').sort_index()


def select_versions(products, version=None):
    """
    One GRS version per Level-1 product: the requested `version`, or the highest one (packaging order:
    ``V3.1.0dev0 < V3.1.0``). Products of the same tile and sensing time but different ESA processing
    times are distinct Level-1 products: both are kept.

    :param products: table of :func:`list_l2a_products`
    :param version: GRS version to keep (e.g. ``'V3.1.0'``), None for the highest of each product
    """
    from packaging.version import InvalidVersion, Version

    if version is not None:
        return products[products['version'] == version]

    def order(v):
        try:
            return Version(str(v).lstrip('Vv'))
        except InvalidVersion:
            return Version('0')

    return (products.assign(_order=products['version'].map(order)).sort_values('_order')
            .reset_index()
            .drop_duplicates(['tile', 'time', 'processing_time'], keep='last')
            .drop(columns='_order')
            .set_index('time').sort_index())


# ---------------------------------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------------------------------

class Profiler:
    """
    Wall time, CPU time and memory of the processing steps of one image.

    A background thread samples the resident memory (RSS) of the process and of its children every
    `interval` seconds; each step, profiled with :meth:`step`, records its wall time, its CPU time (all
    the threads of the process: it exceeds the wall time when the processing runs in parallel) and the
    RSS at its start, peak and end. :meth:`write` stops the sampling and writes the profile as JSON.

    >>> profiler = Profiler('S2A_MSIL1C_..._T31TEJ_....SAFE')
    >>> with profiler.step('L2A'):
    ...     process()
    >>> profiler.write('/data/L2B/profiles/31TEJ/S2A_MSIL1C_..._profile.json')

    :param image: name of the image
    :param interval: sampling interval (s) of the memory
    """

    def __init__(self, image, interval=0.5):
        import psutil

        self.image, self.interval = image, interval
        self.process = psutil.Process()
        self.start = dt.datetime.now()
        self.t0 = time.perf_counter()
        self.steps, self.memory = [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def rss(self):
        """Resident memory (MB) of the process and its children."""
        import psutil

        rss = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                rss += child.memory_info().rss
        return rss / 2**20

    def cpu(self):
        """CPU time (s) of the process and its children, user and system."""
        t = self.process.cpu_times()
        return t.user + t.system + t.children_user + t.children_system

    def _sample(self):
        while not self._stop.is_set():
            self.memory.append((round(time.perf_counter() - self.t0, 2), round(self.rss(), 1)))
            self._stop.wait(self.interval)

    @contextlib.contextmanager
    def step(self, name):
        """Profile the block as the step `name`; its status is ``'failed'`` if it raises."""
        record = dict(step=name, status='done', start_s=round(time.perf_counter() - self.t0, 2),
                      rss_start_mb=round(self.rss(), 1))
        cpu0, t0, n0 = self.cpu(), time.perf_counter(), len(self.memory)
        try:
            yield record
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            end = round(self.rss(), 1)
            wall, cpu = time.perf_counter() - t0, self.cpu() - cpu0
            record.update(wall_s=round(wall, 2), cpu_s=round(cpu, 2),
                          cpu_load=round(cpu / wall, 2) if wall else None,
                          rss_peak_mb=max([end, record['rss_start_mb']] + [m for _, m in self.memory[n0:]]),
                          rss_end_mb=end)
            self.steps.append(record)

    def skip(self, name, status='skipped'):
        """Record the step `name` as not run, with its `status` (e.g. ``'reused'``)."""
        self.steps.append(dict(step=name, status=status))

    def stop(self):
        """Stop the sampling of the memory."""
        self._stop.set()
        self._thread.join()

    def write(self, path, storage_options=None, **info):
        """
        Stop the sampling and write the profile as JSON: image, start, total time, machine (host, CPUs,
        RAM), Python and processor versions, `info`, steps and memory curve (``memory``: time in s since
        the start, RSS in MB).

        :param path: local path or URL of the profile
        :param storage_options: options of the fsspec file system of a URL
        """
        import psutil

        self.stop()
        profile = dict(image=self.image, start=self.start.isoformat(timespec='seconds'),
                       total_s=round(time.perf_counter() - self.t0, 2),
                       host=socket.gethostname(), cpus=os.cpu_count(),
                       ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
                       python=platform.python_version(), versions=_versions(),
                       **info, steps=self.steps, memory=self.memory)
        _write_text(path, json.dumps(profile, indent=1, default=str), storage_options)
        return path


def _versions():
    """Versions of the processors already imported."""
    import sys

    return {name: getattr(sys.modules[name], '__version__', None) for name in ('grs', 'GRSl2bgen')
            if name in sys.modules}


def read_profile(path, storage_options=None):
    """Profile written by :meth:`Profiler.write` (dict), local or URL."""
    return json.loads(_read_text(path, storage_options))


def read_profiles(paths, storage_options=None):
    """
    Table of the steps of profiles: one row per image and step.

    :param paths: profile files, local or URLs (the missing ones are left out)
    :param storage_options: options of the fsspec file system of the URLs
    :return: pandas.DataFrame with ``image``, ``start`` and the records of the steps (``step``,
             ``status``, ``wall_s``, ``cpu_s``, ``cpu_load``, ``rss_start_mb``, ``rss_peak_mb``,
             ``rss_end_mb``...)
    """
    rows = []
    for path in paths:
        if not path or not exists(path, storage_options):
            continue
        profile = read_profile(path, storage_options)
        for step in profile['steps']:
            rows.append(dict(image=image_name(profile['image']), start=profile['start']) | step)
    return pd.DataFrame(rows)


def plot_memory(profile, ax=None, colors=None, storage_options=None):
    """
    Memory curve of a profile, with its steps shaded and their peak memory marked.

    :param profile: profile file (local or URL) or dict (:func:`read_profile`)
    :param ax: matplotlib axes (default: new figure)
    :param colors: colours of the steps (default: L2A blue, L2B orange)
    :param storage_options: options of the fsspec file system of a URL
    :return: the axes
    """
    import matplotlib.pyplot as plt

    if not isinstance(profile, dict):
        profile = read_profile(profile, storage_options)
    colors = colors or {'L2A': 'tab:blue', 'L2B': 'tab:orange'}
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    t, rss = np.array(profile['memory'], dtype=float).reshape(-1, 2).T
    ax.plot(t, rss / 1024, color='k', lw=1.2)
    ends = [step['start_s'] + step['wall_s'] for step in profile['steps'] if 'wall_s' in step]
    tmax = max([t.max() if len(t) else 0, profile.get('total_s', 0)] + ends) or 1
    for step in profile['steps']:
        if step['status'] != 'done':
            continue
        start, end = step['start_s'], step['start_s'] + step['wall_s']
        color = colors.get(step['step'], 'grey')
        ax.axvspan(start, end, color=color, alpha=0.15)
        ax.text((start + end) / 2, 1.01,
                f"{step['step']}: {step['wall_s']:.0f} s, peak {step['rss_peak_mb'] / 1024:.1f} GB",
                transform=ax.get_xaxis_transform(), ha='center', va='bottom')
        ax.axhline(step['rss_peak_mb'] / 1024, xmin=start / tmax, xmax=end / tmax, color=color, ls='--')
    ax.set(xlabel='time since the start of the image (s)', ylabel='memory (RSS, GB)', xlim=(0, tmax))
    ax.set_title(image_name(profile['image']), pad=22)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------------------------------

def grs_version():
    """Version of the installed GRS (imports ``grs``)."""
    import grs

    return grs.__version__


def run_grs(level1, l2a, cams_file=None, dem_file=None, resolution=20, aot_resolution=None,
            opac_model=None, scale_aot=1):
    """
    GRS: Level-1 image -> L2A product, in its own folder with the log ``log_file.log``.

    The product folder is emptied first; on failure it is removed (so that the image is retried) and
    the exception raised again.

    :param level1: Level-1 image folder
    :param l2a: L2A product: ``<folder>/<name>.zarr`` (GRS 3.x) or ``<folder>/<name>.nc`` (GRS 2.x, or
                GRS 3.x with a NetCDF output)
    :param cams_file: CAMS file (GRS 3.x: None to use the CAMS archive of its configuration)
    :param dem_file: DEM GeoTIFF of the tile, or None
    :param resolution: pixel size (m)
    :param aot_resolution: pixel size (m) of the coarse grid of the aerosol retrieval, multiple of
                           `resolution` (GRS 3.x; None: configuration of GRS)
    :param opac_model: aerosol model (None: from CAMS)
    :param scale_aot: scaling factor of the CAMS AOD
    :return: ``'no_water'`` when GRS stopped on an image without water and wrote the marker
             :func:`no_water_marker` instead of the product (GRS 3.1.1 or later), else ``'done'``
    """
    import grs

    outdir = os.path.dirname(l2a)
    remove_output(outdir)
    os.makedirs(outdir)
    # copy of the processing messages in the product folder
    handler = logging.FileHandler(opj(outdir, 'log_file.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    process = None
    try:
        options = dict(cams_file=cams_file, dem_file=dem_file, resolution=resolution,
                       opac_model=opac_model, scale_aot=scale_aot)
        if aot_resolution is not None:
            options['aot_resolution'] = aot_resolution
        # GRS 3.x writes to the product path (format from its extension), GRS 2.x to the folder
        odir = l2a if int(grs.__version__.split('.')[0]) >= 3 else outdir
        process = grs.Process()
        process.execute(level1, odir=odir, **options)
        process.write_output()      # GRS 3.1.1: marker file instead of the product for an image without water
        status = 'no_water' if getattr(process, 'no_water', False) else 'done'
    except Exception:
        logging.exception(f'L2A processing failed: {level1}')
        remove_output(outdir)
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        process = None
        gc.collect()
    return status


def write_l2a_item(l2a):
    """
    STAC Item of an L2A product, ``<name>.json`` next to it (:func:`grstbx.stac.product_item`: content of
    the Items of GRSl2bgen), written last. GRS does not write Items: this one lets the catalogue of the
    products be built from the Items alone, in a bucket (:func:`grstbx.stac.build_catalog`). As with
    GRSl2bgen, the assets are relative to the Item, so that the product folder can be moved or uploaded.

    :param l2a: local L2A product (Zarr store or NetCDF file)
    :return: path of the Item
    """
    from .stac import product_item

    item = product_item(l2a, 'L2A')
    folder = os.path.dirname(os.path.abspath(l2a))
    for asset in item['assets'].values():
        asset['href'] = './' + os.path.relpath(asset['href'], folder)
    path = item_path(l2a)
    tmp = opj(os.path.dirname(path), f'.{os.path.basename(path)}.part{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(item, f, indent=2)
    os.replace(tmp, path)
    return path


def run_grsl2bgen(l2a, l2b, stac=True, pyramid=False, persist_input=False, level=0, l2a_href=None):
    """
    GRSl2bgen: L2A product -> L2B product, in its own folder with the log ``<name>.log`` (written by
    ``GRSl2bgen.logger.RunLogger``, with the resources of the run at its end) and, last, the STAC Item
    ``<name>.json``.

    On failure, the L2B product is removed and the exception raised again (its traceback is in the log).

    :param l2a: L2A product: Zarr store, NetCDF file, or GRS product folder
    :param l2b: L2B product ``<folder>/<name>.nc`` (or ``.zarr``)
    :param stac: write the STAC Item
    :param pyramid: for a ``.zarr`` output, write a multiscale (pyramid) store
    :param persist_input: keep the masked L2A raster in memory (faster, needs more RAM)
    :param level: pyramid level read from a Zarr L2A (0: full resolution)
    :param l2a_href: href of the ``derived_from`` link of the Item to the L2A product (default: `l2a`),
                     e.g. its URL in a bucket when `l2a` is a local copy
    """
    import GRSl2bgen
    from GRSl2bgen.logger import RunLogger
    from GRSl2bgen.stac import write_stac_item

    remove_output(item_path(l2b))      # the Item first: the product is not complete any more
    remove_output(l2b)
    os.makedirs(os.path.dirname(l2b), exist_ok=True)
    log_file = os.path.splitext(l2b)[0] + '.log'
    try:
        with RunLogger(log_file, log_console=True):
            logging.info(f'GRSl2bgen {GRSl2bgen.__version__}: {l2a} -> {l2b}')
            process = GRSl2bgen.Process(l2a, l2b, pyramid=pyramid, persist_input=persist_input, level=level)
            process.run()
            del process
        if stac:
            item_file = write_stac_item(l2b, l2a=l2a, log_file=log_file)
            if l2a_href is not None:
                _relink(item_file, 'derived_from', l2a_href)
    except Exception:
        remove_output(l2b)
        raise
    finally:
        gc.collect()


def _relink(item_file, rel, href):
    """Set the href of the links `rel` of a STAC Item file."""
    with open(item_file) as f:
        item = json.load(f)
    for link in item.get('links', []):
        if link.get('rel') == rel:
            link['href'] = href
    with open(item_file, 'w') as f:
        json.dump(item, f, indent=2)


def no_water_marker(l2a):
    """
    Marker file written by GRS (3.1.1 or later) instead of the L2A product of an image without water (less
    water than its ``min_water_area``): ``<folder>/<name>_no_water.txt`` next to the product
    ``<folder>/<name>.zarr``. The image is then not processed again, as with ``grs --no_clobber``.
    """
    return os.path.splitext(str(l2a).rstrip('/'))[0] + '_no_water.txt'


def _l2a_marker(l2a):
    """File written by GRS once the L2A product holds its full-resolution mask."""
    if str(l2a).endswith('.nc'):
        return l2a
    return opj(zarr_store(l2a), '0', 'mask', 'zarr.json')


class Batch:
    """
    Batch processing, image by image, from Level-1 images to GRS L2A products and, if `l2b_dir` is
    given, to GRSl2bgen L2B products; or from existing L2A products to L2B products.

    The products are written to the trees ``<l2a_dir>/<tile>/YYYY/MM/DD/<name>/<name><l2a_ext>`` and
    ``<l2b_dir>/<tile>/YYYY/MM/DD/<name>/<name><l2b_ext>``, named as with the ``grs`` and ``GRSl2bgen``
    command lines, and the profile of each image to ``<profile_dir>/<tile>/<image>_profile.json``.

    :meth:`plan_level1` and :meth:`plan_l2a` give the state of the products of each image
    (``l2a_done``, ``l2b_done``, see :func:`l2a_complete` and :func:`l2b_complete`) and the steps to run
    (``run_l2a``, ``run_l2b``): GRS for the images without complete L2A product, GRSl2bgen for those
    without complete L2B product, an existing L2A product being reused. :meth:`run` processes them.

    The inputs and the roots of the outputs can be URLs of a bucket (``s3://bucket/L2A_zarr``): the
    inputs of each image are downloaded to `work_dir`, processed there, and the product folders uploaded
    to the bucket (see :mod:`grstbx.batch`).

    :param l2a_dir: root folder of the L2A products, local or URL of a bucket
    :param l2b_dir: root folder of the L2B products, local or URL (None: L2A products only)
    :param cams_file_template: CAMS file, formatted with the acquisition ``date`` (e.g.
                               ``'/data/cams/cams_forecast_{date:%Y-%m}.nc'``); None: CAMS archive of
                               the configuration of GRS (GRS 3.x)
    :param dem_file_template: DEM GeoTIFF of the tile, formatted with ``tile``; None or missing file:
                              no DEM
    :param resolution: pixel size (m) of the L2A products
    :param aot_resolution: GRS 3.x: pixel size (m) of the grid of the aerosol retrieval (None:
                           configuration of GRS)
    :param l2a_ext: ``'.zarr'`` or ``'.nc'`` (None: ``.zarr`` with GRS 3.x, ``.nc`` with GRS 2.x)
    :param max_cloud_cover: images at least this cloudy (0-1) are not processed by GRS
    :param opac_model: aerosol model of GRS (None: from CAMS)
    :param scale_aot: scaling factor of the CAMS AOD
    :param levname: level name of the L2A products
    :param suffix: suffix of the L2A names (None: ``_V<grs version>``)
    :param l2b_ext: ``'.nc'`` or ``'.zarr'``
    :param pyramid: for a ``.zarr`` L2B, write a multiscale (pyramid) store
    :param stac: write the STAC Item of each L2B product (an L2B product is then complete when its Item
                 exists)
    :param l2a_stac: write the STAC Item of each L2A product computed (:func:`write_l2a_item`), needed to
                     build the catalogue of the products from a bucket
    :param persist_input: keep the masked L2A raster in memory in GRSl2bgen
    :param level: pyramid level of the Zarr L2A products read by GRSl2bgen
    :param keep_l2a: keep the L2A product once its L2B product is written
    :param overwrite_l2a: recompute the existing L2A products (and their L2B products)
    :param overwrite_l2b: recompute the existing L2B products, from the existing L2A products
    :param profile_dir: folder of the profiles, local or URL (default: ``<l2b_dir or l2a_dir>/profiles``;
                        False: no profile)
    :param memory_interval: sampling interval (s) of the memory curve of the profiles
    :param version: GRS version (default: version of the installed ``grs``, imported when needed)
    :param work_dir: local folder of the copies of the inputs and outputs in a bucket, one image at a
                     time (default: a temporary folder; a full Sentinel-2 tile needs a few GB)
    :param storage_options: options of the fsspec file system of the URLs (default: credentials and
                            endpoint from the environment, e.g. ``AWS_PROFILE``)
    """

    def __init__(self, l2a_dir=None, l2b_dir=None, cams_file_template=None, dem_file_template=None,
                 resolution=20, aot_resolution=None, l2a_ext=None, max_cloud_cover=1., opac_model=None,
                 scale_aot=1, levname='L2AGRS', suffix=None, l2b_ext='.nc', pyramid=False, stac=True,
                 l2a_stac=False, persist_input=False, level=0, keep_l2a=True, overwrite_l2a=False,
                 overwrite_l2b=False, profile_dir=None, memory_interval=0.5, version=None, work_dir=None,
                 storage_options=None):
        self.l2a_dir, self.l2b_dir = l2a_dir, l2b_dir
        self.cams_file_template, self.dem_file_template = cams_file_template, dem_file_template
        self.resolution, self.aot_resolution = resolution, aot_resolution
        self.max_cloud_cover, self.opac_model, self.scale_aot = max_cloud_cover, opac_model, scale_aot
        self.levname = levname
        self.l2b_ext, self.pyramid, self.stac, self.l2a_stac = l2b_ext, pyramid, stac, l2a_stac
        self.persist_input, self.level = persist_input, level
        self.keep_l2a, self.overwrite_l2a, self.overwrite_l2b = keep_l2a, overwrite_l2a, overwrite_l2b
        if profile_dir is None:
            root = l2b_dir or l2a_dir
            profile_dir = opj(root, 'profiles') if root else False
        self.profile_dir, self.memory_interval = profile_dir, memory_interval
        self._version, self._suffix, self._l2a_ext = version, suffix, l2a_ext
        self._work_dir, self.storage_options = work_dir, storage_options

    @property
    def work_dir(self):
        """Local folder of the copies of the inputs and outputs in a bucket."""
        if self._work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix='grstbx_batch_')
        return self._work_dir

    def _local(self, path, kind):
        """Local copy of a product (or input) of a bucket: ``<work_dir>/<kind>/<folder>/<name>``; local
        paths are unchanged."""
        if not is_url(path):
            return path
        path = str(path).rstrip('/')
        return opj(self.work_dir, kind, os.path.basename(os.path.dirname(path)), os.path.basename(path))

    # -- naming ---------------------------------------------------------------------------------------

    @property
    def version(self):
        """GRS version (of the installed ``grs`` if not given)."""
        if self._version is None:
            self._version = grs_version()
        return self._version

    @property
    def suffix(self):
        """Suffix of the L2A names."""
        return self._suffix if self._suffix is not None else f'_V{self.version}'

    @property
    def l2a_ext(self):
        """Extension of the L2A products."""
        if self._l2a_ext is None:
            return '.zarr' if int(str(self.version).lstrip('Vv').split('.')[0]) >= 3 else '.nc'
        return self._l2a_ext

    def l2a_path(self, level1_name, tile, date):
        """L2A product of a Level-1 image: ``<l2a_dir>/<tile>/YYYY/MM/DD/<name>/<name><l2a_ext>``."""
        name = l2a_name(level1_name, self.levname, self.suffix)
        return opj(self.l2a_dir, tile, f'{pd.Timestamp(date):%Y/%m/%d}', name, name + self.l2a_ext)

    def l2b_path(self, l2a, tile, date):
        """L2B product of an L2A product: ``<l2b_dir>/<tile>/YYYY/MM/DD/<name>/<name><l2b_ext>``."""
        name = l2b_name(l2a)
        return opj(self.l2b_dir, tile, f'{pd.Timestamp(date):%Y/%m/%d}', name, name + self.l2b_ext)

    def profile_path(self, image, tile):
        """Profile of an image: ``<profile_dir>/<tile>/<image>_profile.json`` (None without profile)."""
        if not self.profile_dir:
            return None
        return opj(self.profile_dir, tile, image_name(image) + '_profile.json')

    # -- plans ----------------------------------------------------------------------------------------

    def _steps(self, plan):
        """Steps to run from the state of the products."""
        if self.l2b_dir is None:
            plan['run_l2b'] = False
            plan['run_l2a'] = ~plan['l2a_done'] | self.overwrite_l2a
        else:
            plan['run_l2b'] = ~plan['l2b_done'] | self.overwrite_l2b | self.overwrite_l2a
            plan['run_l2a'] = plan['run_l2b'] & (~plan['l2a_done'] | self.overwrite_l2a)
        if 'no_water' in plan and not self.overwrite_l2a:      # images without water: nothing to compute
            plan['run_l2a'] &= ~plan['no_water']
            plan['run_l2b'] &= ~plan['no_water']
        return plan

    def plan_level1(self, images):
        """
        Products, state and steps to run of Level-1 images.

        :param images: table of :func:`list_images` (index: acquisition date; ``tile``, ``name``,
                       ``path``, ``cloud_cover``)
        :return: copy of `images` with the columns ``cams_file``, ``dem_file``, ``l2a``, ``l2b`` (if
                 `l2b_dir`), ``profile``, ``cams_ok``, ``cloudy``, ``no_water`` (marker of GRS for an image
                 without water, :func:`no_water_marker`: not processed again unless `overwrite_l2a`),
                 ``l2a_done``, ``l2b_done``, ``run_l2a`` and ``run_l2b``
        """
        plan = images.copy()
        dates = list(plan.index)
        if self.cams_file_template is None:
            plan['cams_file'] = None
        else:
            plan['cams_file'] = [self.cams_file_template.format(date=date) for date in dates]
        plan['dem_file'] = [dem_path(self.dem_file_template, tile) for tile in plan['tile']]
        plan['l2a'] = [self.l2a_path(name, tile, date)
                       for name, tile, date in zip(plan['name'], plan['tile'], dates)]
        if self.l2b_dir is not None:
            plan['l2b'] = [self.l2b_path(l2a, tile, date)
                           for l2a, tile, date in zip(plan['l2a'], plan['tile'], dates)]
        plan['profile'] = [self.profile_path(name, tile) for name, tile in zip(plan['name'], plan['tile'])]
        plan['cams_ok'] = [file is None or os.path.exists(file) for file in plan['cams_file']]
        plan['cloudy'] = plan['cloud_cover'] >= self.max_cloud_cover
        plan['no_water'] = [exists(no_water_marker(l2a), self.storage_options) for l2a in plan['l2a']]
        plan['l2a_done'] = plan['l2a'].map(self._l2a_complete).astype(bool)
        plan['l2b_done'] = plan['l2b'].map(self._l2b_complete).astype(bool) if self.l2b_dir is not None else False
        return self._steps(plan)

    def _l2a_complete(self, path):
        return l2a_complete(path, storage_options=self.storage_options)

    def _l2b_complete(self, path):
        return l2b_complete(path, self.stac, self.storage_options)

    def plan_l2a(self, products):
        """
        L2B products, state and steps to run of existing L2A products.

        :param products: table of :func:`list_l2a_products` (index: sensing time; ``tile``, ``path``,
                         and ``complete`` if checked)
        :return: copy of `products` with the columns ``name``, ``l2a``, ``l2b``, ``profile``,
                 ``l2a_done``, ``l2b_done``, ``run_l2a`` (False) and ``run_l2b``
        """
        if self.l2b_dir is None:
            raise ValueError('l2b_dir is needed to plan L2B products')
        plan = products.copy()
        plan['name'] = plan['path'].map(product_name)
        plan['l2a'] = plan['path']
        plan['l2b'] = [self.l2b_path(l2a, tile, date)
                       for l2a, tile, date in zip(plan['l2a'], plan['tile'], plan.index)]
        plan['profile'] = [self.profile_path(name, tile) for name, tile in zip(plan['name'], plan['tile'])]
        plan['cams_ok'], plan['cloudy'], plan['no_water'] = True, False, False
        plan['l2a_done'] = (plan['complete'] if 'complete' in plan
                            else plan['l2a'].map(self._l2a_complete)).astype(bool)
        plan['l2b_done'] = plan['l2b'].map(self._l2b_complete).astype(bool)
        plan = self._steps(plan)
        # an incomplete L2A product cannot be processed
        plan['run_l2b'] &= plan['l2a_done']
        plan['run_l2a'] = False
        return plan

    @staticmethod
    def describe(plan):
        """Counts of the products done and to compute (text)."""
        blocked = plan['run_l2a'] & (~plan['cams_ok'] | plan['cloudy'])
        if 'cloud_cover' in plan:       # Level-1 inputs
            lines = [f"L2A: {plan['l2a_done'].sum()} done, {(plan['run_l2a'] & ~blocked).sum()} to compute, "
                     f"{(blocked & plan['cloudy']).sum()} too cloudy, {(blocked & ~plan['cams_ok']).sum()} "
                     'without CAMS file']
            if 'no_water' in plan:
                lines[0] += f", {(plan['no_water'] & ~plan['run_l2a']).sum()} without water"
        else:                           # L2A inputs
            lines = [f"L2A: {plan['l2a_done'].sum()} complete, {(~plan['l2a_done']).sum()} incomplete"]
        if 'l2b' in plan:
            lines.append(f"L2B: {plan['l2b_done'].sum()} done, {(plan['run_l2b'] & ~blocked).sum()} to compute")
        return '\n'.join(lines)

    # -- processing -----------------------------------------------------------------------------------

    def process(self, row):
        """
        Steps of one image of a plan: L2A (GRS) if ``run_l2a``, else the existing L2A product is reused,
        then L2B (GRSl2bgen) if ``run_l2b``. A failing step is logged, its partial output removed, and the
        L2B step is not run after a failed L2A step. The profile of the image is written whatever the
        outcome.

        :param row: row of a plan (pandas.Series)
        :return: record of the summary: ``tile``, ``name``, ``status`` (``done``, ``skipped``,
                 ``no_cams``, ``cloudy``, ``no_water``, ``L2A failed``, ``L2B failed``), ``l2a`` and ``l2b``
                 (``done``, ``reused``, ``no_water`` or empty), ``error``, ``duration_s``. ``no_water``: GRS
                 found no water in the image (now, or in a previous run: marker :func:`no_water_marker`)
        """
        record = dict(tile=row['tile'], name=row['name'], status='done', l2a='', l2b='', error='',
                      duration_s=0.)
        if not (row['run_l2a'] or row['run_l2b']):
            return record | dict(status='no_water' if row.get('no_water', False) else 'skipped')
        if row['run_l2a'] and not row['cams_ok']:
            return record | dict(status='no_cams', error=f"missing {row['cams_file']}")
        if row['run_l2a'] and row['cloudy']:
            return record | dict(status='cloudy')

        so = self.storage_options
        profiler = Profiler(row['name'], interval=self.memory_interval)
        # local paths of the products: copies in work_dir of the products of a bucket, removed at the end
        l2a = self._local(row['l2a'], 'L2A')
        staged = []
        try:
            if row['run_l2a']:
                with profiler.step('L2A') as step:
                    level1 = row['path']
                    if is_url(level1):
                        level1 = download(level1, self._local(level1, 'L1'), so)
                        staged.append(level1)
                    if is_url(row['l2a']):
                        staged.append(os.path.dirname(l2a))
                    status = run_grs(level1, l2a, cams_file=row['cams_file'], dem_file=row['dem_file'],
                                     resolution=self.resolution, aot_resolution=self.aot_resolution,
                                     opac_model=self.opac_model, scale_aot=self.scale_aot)
                    if status == 'no_water':        # marker file instead of the product: no Item, no L2B
                        step['status'] = 'no water'
                        if is_url(row['l2a']):
                            upload(no_water_marker(l2a), no_water_marker(row['l2a']), so)
                    else:
                        if self.l2a_stac:
                            write_l2a_item(l2a)
                        if is_url(row['l2a']):
                            upload(os.path.dirname(l2a), os.path.dirname(row['l2a']), so, last=item_path(l2a))
                if status == 'no_water':
                    if row['run_l2b']:
                        profiler.skip('L2B', 'not run')
                    record |= dict(status='no_water', l2a='no_water')
                    return record
                record['l2a'] = 'done'
            else:
                profiler.skip('L2A', 'reused')
                record['l2a'] = 'reused'
            if row['run_l2b']:
                with profiler.step('L2B'):
                    if is_url(row['l2a']) and record['l2a'] == 'reused':
                        download(row['l2a'], l2a, so)
                        staged.append(os.path.dirname(l2a))
                    l2b = self._local(row['l2b'], 'L2B')
                    if is_url(row['l2b']):
                        staged.append(os.path.dirname(l2b))
                    run_grsl2bgen(l2a, l2b, stac=self.stac, pyramid=self.pyramid,
                                  persist_input=self.persist_input, level=self.level,
                                  l2a_href=row['l2a'] if is_url(row['l2a']) else None)
                    if is_url(row['l2b']):
                        upload(os.path.dirname(l2b), os.path.dirname(row['l2b']), so, last=item_path(l2b))
                record['l2b'] = 'done'
                if not self.keep_l2a:
                    remove_output(os.path.dirname(row['l2a']), so)
        except Exception as e:
            step = 'L2B' if record['l2a'] else 'L2A'
            if step == 'L2A' and row['run_l2b']:
                profiler.skip('L2B', 'not run')
            record |= dict(status=f'{step} failed', error=f'{type(e).__name__}: {e}')
        finally:
            for path in staged:
                remove_output(path)
            profiler.stop()
            if row['profile']:
                level1 = row['path'] if 'cloud_cover' in row.index else None
                profiler.write(row['profile'], storage_options=so, l1=level1, l2a=row['l2a'],
                               l2b=row.get('l2b'), resolution=self.resolution, status=record['status'])
            record['duration_s'] = round(time.perf_counter() - profiler.t0, 1)
        return record

    def run(self, plan, verbose=True):
        """
        Process the images of a plan, one after the other.

        :param plan: table of :meth:`plan_level1` or :meth:`plan_l2a`
        :param verbose: print a line per image (status, name, duration)
        :return: summary table, one row per image (index: date)
        """
        records = []
        for i, (date, row) in enumerate(plan.iterrows(), start=1):
            record = self.process(row)
            records.append(dict(date=date) | record)
            if verbose:
                print(f"[{i}/{len(plan)}] {record['status']:10s} {row['name']} ({record['duration_s']:.0f} s)")
        columns = ['date', 'tile', 'name', 'status', 'l2a', 'l2b', 'error', 'duration_s']
        return pd.DataFrame(records, columns=columns).set_index('date')

    def save_summary(self, summary, prefix='batch'):
        """Write a summary to ``<l2b_dir or l2a_dir>/<prefix>_<time>.csv`` (local or URL); return the path."""
        root = self.l2b_dir or self.l2a_dir
        path = opj(root, f'{prefix}_{dt.datetime.now():%Y%m%dT%H%M%S}.csv')
        _write_text(path, summary.to_csv(), self.storage_options)
        return path

    # -- command line ---------------------------------------------------------------------------------

    def commands(self, plan, profile=True):
        """
        Shell commands of a plan, one line per image with steps to run: ``grs`` if ``run_l2a``, then
        ``GRSl2bgen`` if ``run_l2b`` and the L2A product exists, each run through GNU ``time -v`` if
        `profile` (report ``<profile>_L2A.txt``, ``<profile>_L2B.txt``). Unless overwriting, the products
        completed in the meantime are skipped (``grs --no_clobber``, STAC Item of the L2B product). The
        ``grs`` command line writes Zarr products with GRS 3.x and NetCDF products with GRS 2.x, whatever
        `l2a_ext`.

        The command lines work on local folders: for products in a bucket, use :meth:`run`, or mount the
        bucket as a folder (``rclone mount``).

        :param plan: table of :meth:`plan_level1` or :meth:`plan_l2a`
        :param profile: profile each step with GNU ``time -v``
        :return: list of shell command lines
        """
        urls = [p for p in (self.l2a_dir, self.l2b_dir, self.profile_dir or None) if p and is_url(p)]
        if urls or plan['path'].map(is_url).any():
            raise ValueError('the command lines need local folders, not URLs: use Batch.run, or mount the bucket '
                             f'(rclone mount) {urls}')
        commands = []
        for date, row in plan.iterrows():
            stem = os.path.splitext(row['profile'])[0] if profile and row['profile'] else None
            timer = (lambda step: f'/usr/bin/time -v -o {stem}_{step}.txt ') if stem else (lambda step: '')
            steps = []
            if row['run_l2a']:
                if not row['cams_ok'] or row['cloudy']:
                    continue
                cmd = (f"grs {row['path']} --odir {opj(self.l2a_dir, row['tile'], f'{date:%Y/%m/%d}')} "
                       f"--resolution {self.resolution} --max_cloud_cover {self.max_cloud_cover} "
                       f"--scale_aot {self.scale_aot} --levname {self.levname}")
                if not self.overwrite_l2a:
                    cmd += ' --no_clobber'
                if row['cams_file']:
                    cmd += f" --cams_file {row['cams_file']}"
                if self.aot_resolution is not None:
                    cmd += f' --aot_resolution {self.aot_resolution}'
                if row['dem_file']:
                    cmd += f" --dem_file {row['dem_file']}"
                if self.opac_model:
                    cmd += f' --opac_model {self.opac_model}'
                if self._suffix is not None:
                    cmd += f' --suffix {self._suffix}'
                steps.append(timer('L2A') + cmd)
            if row['run_l2b']:
                # written to <folder>/<name>/ by --product_dir
                out = opj(os.path.dirname(os.path.dirname(row['l2b'])), os.path.basename(row['l2b']))
                l2a = os.path.dirname(row['l2a']) if 'cloud_cover' in plan else row['l2a']
                cmd = f'GRSl2bgen {l2a} -o {out} --product_dir' + (' --stac' if self.stac else '')
                # grs exits with 0 when it fails and 255 when the product exists (--no_clobber): the
                # L2B step runs if the L2A product exists
                cmd = f'[ -e {_l2a_marker(row["l2a"])} ] && {timer("L2B")}{cmd}'
                steps.append(cmd)
                if not (self.overwrite_l2a or self.overwrite_l2b):
                    guard = f"[ -e {item_path(row['l2b']) if self.stac else row['l2b']} ] || "
                else:
                    guard = ''
            else:
                guard = ''
            if not steps:
                continue
            mkdir = f'mkdir -p {os.path.dirname(stem)}; ' if stem else ''
            commands.append(f"{guard}{{ {mkdir}{'; '.join(steps)}; }}")
        return commands
