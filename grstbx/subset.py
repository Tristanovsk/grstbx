"""
Crop GRS L2A products to an area of interest (AOI) and export the subsets.

The subsets are written with the exporters of GRS, so that they keep the layout and the packing of
the products of the processor and can be reopened with :class:`~grstbx.driver.L2grs`:

- NetCDF: ``<name>/<name>.nc`` (main) and ``<name>/<name>_anc.nc`` (ancillary);
- Zarr: ``<name>.zarr`` (full resolution in group '0', pyramid overviews, group 'ancillary').

Example
-------
>>> import grstbx
>>> product, ancillary = grstbx.open_l2a('S2A_MSIL2AGRS_..._V3.0.1dev.zarr')
>>> aoi = grstbx.aoi_from_box(-2.25, 47.22, width=15000, height=12000)
>>> subset, subset_anc = grstbx.crop_l2a(product, aoi, ancillary)
>>> grstbx.export_l2a(subset.load(), '/out/scene_site/scene_site.nc', ancillary=subset_anc)

or in one call, e.g. for a series of images:

>>> grstbx.crop_and_export(file, aoi, grstbx.subset_path(file, '/out', 'site'))

The export needs the ``grs`` package >= 3.0.1 (imported only when exporting).
"""

import os
import json
import functools

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # activate the rio accessor
from rasterio.features import geometry_mask

from .driver import L2grs, is_zarr

# bit of the 'nodata' flag in the GRS bitmask
NODATA_BIT = 0


def open_l2a(path, level=0):
    """
    Open a GRS L2A product lazily, with its ancillary data.

    :param path: ``*.zarr`` store or NetCDF product folder ``<name>/`` (see
                 :meth:`L2grs.load_l2a_image <grstbx.driver.L2grs.load_l2a_image>`)
    :param level: pyramid level for multiscale Zarr stores (0: full resolution, -1: coarsest)
    :return: (product, ancillary) xarray.Datasets; ancillary is None when absent
    """
    return L2grs([path], level=level).load_l2a_image(path)


def aoi_from_box(lon, lat, width, height):
    """
    Rectangular AOI centred on a point.

    :param lon: longitude of the centre (deg)
    :param lat: latitude of the centre (deg)
    :param width: width of the box (m)
    :param height: height of the box (m)
    :return: geopandas.GeoDataFrame (EPSG:4326) with one polygon
    """
    from .utils import SpatioTemp

    box = SpatioTemp().wktbox(lon, lat, width=width, height=height, ellps='WGS84')
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt([box]), crs=4326)


def crop_l2a(product, aoi, ancillary=None, mask_outside=True):
    """
    Crop an L2A product to an AOI (lazy: only the chunks intersecting the AOI are read when the
    result is loaded).

    The product is cut to the bounding box of the AOI. With ``mask_outside=True``, the pixels of the
    bounding box outside the polygon(s) are then flagged as missing, following the GRS conventions:
    float variables set to NaN, ``nodata`` bit (0) of ``flags`` raised, ``mask`` set to 1. The integer
    variables keep their dtype.

    The ancillary data are cropped to the same bounding box enlarged by one coarse cell, so that they
    can still be interpolated over the whole subset.

    :param product: L2A xarray.Dataset with a CRS (rio accessor)
    :param aoi: geopandas.GeoDataFrame (or GeoSeries), polygon(s) in any CRS
    :param ancillary: ancillary xarray.Dataset (dims ``xc``, ``yc``) or None
    :param mask_outside: mark the pixels outside the polygon(s) as nodata
    :return: (subset, ancillary subset); the ancillary subset is None if ``ancillary`` is None
    """
    aoi = aoi.to_crs(product.rio.crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    subset = product.rio.clip_box(minx, miny, maxx, maxy)

    if mask_outside:
        inside = ~geometry_mask(aoi.geometry, out_shape=(subset.sizes['y'], subset.sizes['x']),
                                transform=subset.rio.transform(recalc=True))
        inside = xr.DataArray(inside, dims=('y', 'x'), coords={'y': subset.y, 'x': subset.x})
        for var in subset.data_vars:
            da = subset[var]
            if not {'x', 'y'} <= set(da.dims):
                continue
            if var == 'flags':
                da = da.where(inside, da | (1 << NODATA_BIT))
            elif var == 'mask':
                da = da.where(inside, 1)
            else:
                da = da.where(inside)
            subset[var] = da.astype(product[var].dtype).assign_attrs(product[var].attrs)

    if ancillary is not None:
        dx, dy = (float(abs(ancillary[c][1] - ancillary[c][0])) for c in ('xc', 'yc'))
        ancillary = ancillary.sel(xc=slice(minx - dx, maxx + dx), yc=slice(maxy + dy, miny - dy))

    return subset, ancillary


def _netcdf_safe(attrs):
    '''Attributes with booleans, None and dictionaries converted to strings (not supported by NetCDF).'''

    def convert(value):
        if isinstance(value, (bool, np.bool_)) or value is None:
            return str(value)
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_, dict)) or v is None
                                                    for v in value):
            return [str(v) for v in value]
        return value

    return {key: convert(value) for key, value in attrs.items()}


def _sanitize(ds):
    ds = ds.copy()
    ds.attrs = _netcdf_safe(ds.attrs)
    for var in ds.variables:
        ds[var].attrs = _netcdf_safe(ds[var].attrs)
    return ds


@functools.cache
def _exporter_class():
    '''Exporter built on ``grs.output.L2aProduct`` (grs is imported here, only when exporting).'''
    try:
        from grs.output import L2aProduct
    except ImportError as err:
        raise ImportError('exporting L2A products needs the grs package >= 3.0.1 '
                          '(https://github.com/Tristanovsk/grs)') from err
    if not hasattr(L2aProduct, '_prepare_export'):
        import grs
        raise ImportError(f'exporting L2A products needs grs >= 3.0.1 (installed: {grs.__version__})')

    class L2aExporter(L2aProduct):
        '''Feed an existing L2A dataset to the GRS writers, without running the processor.'''

        def __init__(self, l2_prod, ancillary, complevel=5):
            # the constructor of L2aProduct builds the product from the processor outputs: not called
            self.l2_prod = _sanitize(l2_prod)
            # the GRS writers always write the ancillary part: empty when there is none
            self.ancillary = _sanitize(xr.Dataset() if ancillary is None else ancillary)
            self.complevel = complevel

        def _prepare_export(self, snap_compliant=False):
            ds, encoding = super()._prepare_export(snap_compliant)
            # variables unknown to the GRS packing (e.g. 'wind') written as float32
            for var in ds.data_vars:
                if var not in encoding and ds[var].dtype.kind == 'f':
                    encoding[var] = {'dtype': 'float32', 'grid_mapping': 'spatial_ref'}
            return ds, encoding

    return L2aExporter


def export_l2a(product, path, ancillary=None, fmt=None, complevel=5, **kwargs):
    """
    Write an L2A dataset (e.g. a subset from :func:`crop_l2a`) with the GRS writers: same layout and
    packing as the products of the processor (``int16`` with scale/offset for ``Rrs``, ``BRDFg``,
    ``aot550``, angles, ``dem``; ``flags`` and ``mask`` unchanged).

    Attributes NetCDF cannot store (booleans, None, dictionaries) are converted to strings.

    :param product: L2A xarray.Dataset; load it first (``.load()``) if it is used again afterwards
    :param path: ``<dir>/<name>.nc`` (NetCDF, ancillary in ``<dir>/<name>_anc.nc``) or
                 ``<dir>/<name>.zarr`` (Zarr store with pyramids); see :func:`subset_path`
    :param ancillary: ancillary xarray.Dataset, or None (an empty ancillary part is then written)
    :param fmt: ``'netcdf'`` or ``'zarr'`` to force the format (default: from the extension of ``path``)
    :param complevel: zlib compression level for NetCDF
    :param kwargs: options of ``grs.output.L2aProduct.export_to_zarr`` (chunks, pyramids, ...)
    :return: path to give to :class:`~grstbx.driver.L2grs`: the Zarr store, or the NetCDF product folder
    """
    _exporter_class()(product, ancillary, complevel=complevel).export(path, fmt=fmt, **kwargs)
    if is_zarr(path) or fmt == 'zarr':
        return path
    return os.path.dirname(path)


def export_rrs_geotiff(product, path, water_only=False):
    """
    Write ``Rrs`` as a compressed multi-band float32 GeoTIFF (one band per wavelength, named
    ``Rrs_<wl>``, NaN for missing pixels), e.g. for GIS software.

    :param product: L2A xarray.Dataset
    :param path: output ``.tif`` file
    :param water_only: keep only the valid water pixels (``mask == 0``)
    :return: path
    """
    rrs = product.Rrs.where(product['mask'] == 0) if water_only else product.Rrs
    rrs = rrs.astype(np.float32).rio.write_nodata(np.nan)
    rrs.attrs = {'long_name': tuple(f'Rrs_{int(wl)}' for wl in rrs.wl.values), 'units': 'sr-1'}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rrs.rio.to_raster(path, compress='deflate', predictor=3, tiled=True)
    return path


def subset_path(file, odir, site, fmt='netcdf'):
    """
    Output path of the subset of ``file``: ``odir/site/<product>_<site>/<product>_<site>.nc`` (NetCDF)
    or ``odir/site/<product>_<site>.zarr`` (Zarr).

    :param file: input product (Zarr store or NetCDF folder)
    :param odir: output directory
    :param site: name of the AOI
    :param fmt: ``'netcdf'`` or ``'zarr'``
    :return: path to give to :func:`export_l2a`
    """
    name = os.path.basename(os.path.normpath(file))
    # strip the format extension only: product names contain dots (e.g. '_V2.2.1')
    for ext in ('.zarr', '.nc'):
        name = name.removesuffix(ext)
    name = name + '_' + site
    if fmt == 'zarr':
        return os.path.join(odir, site, name + '.zarr')
    return os.path.join(odir, site, name, name + '.nc')


def crop_and_export(file, aoi, path, mask_outside=True, overwrite=False, **kwargs):
    """
    Open an L2A product, crop it to the AOI and export the subset (see :func:`crop_l2a` and
    :func:`export_l2a`).

    Example
    -------
    >>> aoi = grstbx.aoi_from_box(-2.25, 47.22, width=15000, height=12000)
    >>> for file in files:
    ...     grstbx.crop_and_export(file, aoi, grstbx.subset_path(file, '/out', 'loire'))

    :param file: input product (Zarr store or NetCDF folder)
    :param aoi: geopandas.GeoDataFrame, polygon(s) in any CRS
    :param path: output path (``.nc`` or ``.zarr``), see :func:`subset_path`
    :param mask_outside: mark the pixels outside the polygon(s) as nodata
    :param overwrite: if False, an existing output is kept and not recomputed
    :param kwargs: passed to :func:`export_l2a`
    :return: path of the exported product to give to :class:`~grstbx.driver.L2grs`
    """
    out = path if is_zarr(path) else os.path.dirname(path)
    if not overwrite and os.path.exists(path):
        return out
    product, ancillary = open_l2a(file)
    subset, subset_anc = crop_l2a(product, aoi, ancillary, mask_outside=mask_outside)
    return export_l2a(subset.load(), path, ancillary=subset_anc, **kwargs)
