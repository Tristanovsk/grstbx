"""
Loading of GRS L2A (remote sensing reflectance) and L2B (water quality parameters)
images into multi-temporal xarray datacubes.
"""

import os
import logging

import numpy as np
import xarray as xr
import rioxarray as rxr  # activate the rio accessor
from affine import Affine

opj = os.path.join


def is_zarr(path):
    """True if ``path`` is a Zarr store (by its '.zarr' extension)."""
    return str(path).rstrip('/').endswith('.zarr')


def zarr_levels(path):
    """
    List the resolution levels of a multiscale (pyramid) Zarr store.

    Pyramids hold one group per level: '0' (full resolution), '1' (2x coarser), '2', ...
    The levels are read from the ``multiscales`` attribute of the root group when present,
    otherwise from the numbered sub-groups.

    :param path: path to the Zarr store (Zarr format 2 or 3)
    :return: list of level group names ordered from full to coarsest resolution;
             empty list for a single-resolution store
    """
    import zarr

    root = zarr.open_group(path, mode='r')
    multiscales = root.attrs.get('multiscales')
    if isinstance(multiscales, dict) and multiscales.get('layout'):
        return [str(level['asset']) for level in multiscales['layout']]
    return sorted((name for name, _ in root.groups() if name.isdigit()), key=int)


def open_zarr_image(path, level=0, chunks={}):
    """
    Open a Zarr image, either single-resolution or multiscale (pyramid).

    For a pyramid, the group of the requested ``level`` is opened and the product
    attributes stored on the root group are added to the dataset (without overwriting
    those of the level). The level opened is stored in the ``pyramid_level`` attribute.

    Example
    -------
    >>> zarr_levels('S2A_L2B.zarr')
    ['0', '1', '2', '3']
    >>> ds = open_zarr_image('S2A_L2B.zarr', level=2)    # 4x coarser than full resolution
    >>> ds = open_zarr_image('S2A_L2B.zarr', level=-1)   # coarsest level

    :param path: path to the Zarr store (Zarr format 2 or 3)
    :param level: pyramid level, as index of the available levels (0: full resolution,
                  negative values count from the coarsest); ignored for a single-resolution store
    :param chunks: dask chunks passed to ``xarray.open_zarr`` (default: chunks of the store)
    :return: xarray.Dataset
    """
    levels = zarr_levels(path)
    if not levels:
        return xr.open_zarr(path, decode_coords='all', chunks=chunks)

    try:
        group = levels[level]
    except IndexError:
        raise ValueError(f'level {level} not available in {path}, levels: {levels}') from None

    import zarr

    ds = xr.open_zarr(path, group=group, decode_coords='all', chunks=chunks)
    root_attrs = {k: v for k, v in zarr.open_group(path, mode='r').attrs.items() if k != 'multiscales'}
    ds.attrs = {**root_attrs, **ds.attrs, 'pyramid_level': int(group)}
    return ds


def open_zarr_ancillary(path, group='ancillary'):
    """
    Open the ancillary data (CAMS fields, gaseous transmittance on a coarse grid, dims ``xc``, ``yc``)
    stored in the group ``group`` of a GRS L2A Zarr store.

    :param path: path to the Zarr store
    :param group: name of the ancillary group
    :return: xarray.Dataset, or None if the store has no such group
    """
    import zarr

    if group not in dict(zarr.open_group(path, mode='r').groups()):
        return None
    return xr.open_zarr(path, group=group, decode_coords='all')


class L2grs():
    """
    Build multi-temporal datacubes from a list of GRS products.

    Example
    -------
    >>> dc = L2grs(files)
    >>> dc.get_l2a_datacube(subset=aoi_geodataframe)
    >>> dc.datacube            # xarray.Dataset with a 'time' dimension

    :param files: list of paths to GRS products (L2A folders, L2B netCDF files or .zarr stores)
    :param level: resolution level opened for multiscale (pyramid) Zarr stores,
                  0: full resolution, 1: 2x coarser, ..., -1: coarsest (see ``open_zarr_image``)
    """

    def __init__(self, files, level=0):
        self.files = files
        self.level = level
        # dask chunk sizes used when opening images
        self.xchunk = 1000
        self.ychunk = 1000
        self.wlchunk = -1

    @property
    def _chunks(self):
        return {'wl': self.wlchunk, 'x': self.xchunk, 'y': self.ychunk}

    def load_l2a_image(self,
                       l2a_path,
                       level=None):
        """
        Open a GRS L2A product.

        Two layouts are supported:
         - ``*.zarr`` store, single-resolution or multiscale (pyramid with groups '0', '1', ...),
           with the ancillary data in the group 'ancillary' when present;
         - folder ``<name>/`` containing ``<name>.nc`` (main) and ``<name>_anc.nc`` (ancillary).

        Products written with the 'beam' metadata profile store one variable per band
        (``Rrs_<wl>``); they are reshaped into a single ``Rrs`` variable with a ``wl`` dimension.

        :param l2a_path: path to the L2A product
        :param level: pyramid level for multiscale Zarr stores (default: ``self.level``)
        :return: (raster, ancillary) xarray.Datasets; ancillary is None for zarr stores without
                 'ancillary' group
        """

        if is_zarr(l2a_path):
            return (open_zarr_image(l2a_path, level=self.level if level is None else level),
                    open_zarr_ancillary(l2a_path))

        basename = os.path.basename(l2a_path.rstrip('/'))
        main_file = opj(l2a_path, basename + '.nc')
        ancillary_file = opj(l2a_path, basename + '_anc.nc')

        raster = xr.open_dataset(main_file,
                                 decode_coords='all',
                                 chunks=self._chunks)
        ancillary = xr.open_dataset(ancillary_file,
                                    decode_coords='all')

        if raster.attrs.get('metadata_profile') != 'beam':
            return raster, ancillary

        # reshape into datacube:
        wls = raster.wl.values

        if 'wl' in raster.dims:
            raster = raster.drop_dims('wl')

        Rrs_vars = ['Rrs_{:d}'.format(int(wl)) for wl in wls]

        Rrs = raster[Rrs_vars].to_array(dim='wl', name='Rrs').chunk(self._chunks)
        Rrs = Rrs.assign_coords({'wl': wls})
        raster = raster.drop_vars(Rrs_vars)
        return xr.merge([raster, Rrs]), ancillary

    def load_l2b_image(self, l2b_path, level=None):
        """
        Open a GRS L2B product (lazy, dask-chunked): netCDF file or Zarr store,
        single-resolution or multiscale (pyramid with groups '0', '1', ...).

        :param l2b_path: path to the L2B product
        :param level: pyramid level for multiscale Zarr stores (default: ``self.level``)
        :return: xarray.Dataset
        """
        if is_zarr(l2b_path):
            return open_zarr_image(l2b_path, level=self.level if level is None else level)
        return xr.open_dataset(l2b_path, decode_coords='all', chunks={'x': self.xchunk, 'y': self.ychunk})

    def subset_xy(self, ds, bbox):
        """
        Crop ``ds`` to the bounding box of ``bbox`` and load the result in memory.

        :param ds: xarray object with a CRS (rio accessor) and descending y coordinates
        :param bbox: geopandas.GeoDataFrame, only the bounds of its first geometry are used
        :return: cropped (and loaded) xarray object
        """

        bbox = bbox.to_crs(epsg=ds.rio.crs.to_epsg())
        minx, miny, maxx, maxy = bbox.bounds.values[0]
        return ds.sel(x=slice(minx, maxx), y=slice(maxy, miny)).load()

    def _finalize_datacube(self, product):
        self.datacube = product
        self.datacube.attrs['start_date'] = str(product.time[0].values)
        self.datacube.attrs['stop_date'] = str(product.time[-1].values)
        self.pixnum = len(self.datacube.x) * len(self.datacube.y)

    def get_l2a_datacube(self,
                         subset=None,
                         reproject=False,
                         nodata_thresh=0.5,
                         epsg_out=3857,
                         FLAG_NAME='flags'):
        """
        Load all L2A ``self.files`` into ``self.datacube`` (concatenated along 'time').

        Per-flag pixel proportions (``flag_<name>`` variables, see ``get_flag_stats``)
        are added to the datacube; images whose ``flag_nodata`` proportion exceeds
        ``nodata_thresh`` are skipped. If no image is kept, ``self.no_product`` is set
        to True and ``self.datacube`` is not created.

        :param subset: geopandas.GeoDataFrame, crop images to its bounding box
        :param reproject: if True, reproject images to ``epsg_out``
        :param nodata_thresh: maximum accepted proportion of nodata pixels
        :param epsg_out: EPSG code used when ``reproject`` is True
        :param FLAG_NAME: name of the bitmask variable
        """

        products = []
        for file in self.files:
            logging.info(f'loading l2a image: {file}')
            product, anc = self.load_l2a_image(file)

            # add mean solar angles:
            for attribute in ['mean_solar_azimuth', 'mean_solar_zenith_angle']:
                if attribute in product.attrs:
                    product[attribute] = product.attrs[attribute]

            if subset is not None:
                logging.info('subsetting...')
                product = self.subset_xy(product, subset)

            if reproject:
                logging.info('reprojecting...')
                product = product.rio.reproject(epsg_out)
                self.epsg = product.rio.crs.to_epsg()

            # get flag statistics and discard images with too many nodata pixels
            logging.info('computing flags statistics')
            flag_stats = self.get_flag_stats(product[FLAG_NAME].expand_dims('time'))
            if 'flag_nodata' in flag_stats and flag_stats.flag_nodata.values > nodata_thresh:
                continue

            products.append(xr.merge([product, flag_stats]))

        if len(products) == 0:
            self.no_product = True
            return

        logging.info('concatenate rasters')
        product = xr.concat(products, dim='time', data_vars='all')
        # keep only one date for dem
        if 'dem' in product:
            product['dem'] = product.dem.isel(time=0)

        self._finalize_datacube(product)

    def get_l2b_datacube(self,
                         subset=None,
                         reproject=False,
                         epsg_out=3857,
                         var='Chla_OC2nasa',
                         var_novalid='central_wavelength'):
        """
        Load all L2B ``self.files`` into ``self.datacube`` (concatenated and sorted along 'time').

        A ``valid_pix_prop`` variable (proportion of valid pixels of ``var``) is added;
        images without any valid pixel are skipped. If no image is kept,
        ``self.no_product`` is set to True.

        :param subset: geopandas.GeoDataFrame, crop images to its bounding box
        :param reproject: if True, reproject images to ``epsg_out``
        :param epsg_out: EPSG code used when ``reproject`` is True
        :param var: variable used to count valid pixels
        :param var_novalid: variable dropped before concatenation (incompatible across dates)
        """
        products = []
        for file in self.files:

            product = self.load_l2b_image(file)

            if var_novalid in product:
                product = product.drop_vars(var_novalid)

            if subset is not None:
                product = self.subset_xy(product, subset)

            if reproject:
                product = product.rio.reproject(epsg_out)
                self.epsg = product.rio.crs.to_epsg()

            # check valid pixels:
            Npix_tot = product.sizes['x'] * product.sizes['y']
            Npix_valid = int(product[var].count().compute())
            if Npix_valid == 0:
                continue
            product['valid_pix_prop'] = Npix_valid / Npix_tot
            product['valid_pix_prop'].attrs['description'] = 'Proportion of valid pixels for ' + var \
                                                             + ' within the image raster'

            products.append(product)

        if len(products) == 0:
            self.no_product = True
            return

        self._finalize_datacube(xr.concat(products, dim='time', data_vars='all').sortby('time'))

    @staticmethod
    def get_flag_stats(raster):
        '''
        Compute, for each flag and each time, the proportion of pixels where the flag is raised.

        :param raster: bitmask xarray.DataArray with a 'time' dimension and a
                       'flag_names' attribute (bit i <-> flag_names[i])
        :return: xarray.Dataset with one ``flag_<name>`` variable (dims: time) per named flag
        '''

        spatial_dims = [dim for dim in raster.dims if dim != 'time']
        flags = raster.compute()
        npix = flags.count(dim=spatial_dims)

        flag_stats = {}
        for bit, flag_name in enumerate(flags.attrs['flag_names']):
            if flag_name in ('None', ''):
                continue
            raised = (flags & (1 << bit)) != 0
            flag_stats['flag_' + flag_name] = (raised.sum(dim=spatial_dims) / npix).astype(float)

        return xr.Dataset(flag_stats).reset_coords(drop=True).assign_coords({'time': raster.time.values})

    def reshape_raster(self, bands=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                    'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                    'Rrs_B8A'],
                       data_vars=['SZA', 'AZI', 'VZA', 'shade', 'BRDFg'],
                       from_datacube=False
                       ):
        """
        Legacy (GRS v1): stack per-band variables into a single ``Rrs`` variable with a
        ``wl`` dimension (taken from each band's 'wavelength' attribute). The result,
        merged with ``data_vars``, is stored in ``self.raster``.

        :param bands: per-band variables to stack
        :param data_vars: other variables to keep
        :param from_datacube: use ``self.datacube`` as input instead of ``self.raster``
        """
        p_ = self.datacube if from_datacube else self.raster
        wl = [band.attrs['wavelength'] for band in p_[bands].values()]

        Rrs = p_[bands].to_array(dim='wl', name='Rrs').assign_coords(wl=wl).chunk({'wl': 1})

        # merge to keep flags
        self.raster = xr.merge([Rrs, p_[data_vars]]).chunk({'time': 1})

    def reproject_data_vars(self, epsg=3857,
                            data_vars=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                       'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                       'Rrs_B8A', 'SZA', 'AZI', 'VZA', 'shade', 'BRDFg'],
                            no_data=np.nan
                            ):
        """
        Reproject ``data_vars`` of ``self.datacube`` to ``epsg``; the result is stored in ``self.raster``.

        :param epsg: output EPSG code
        :param data_vars: variables to reproject
        :param no_data: nodata value written before reprojection
        """

        # set no_data value before reprojection
        for var in data_vars:
            # TODO handle different nodata value depending on dtype (int, float, str...)
            self.datacube[var].rio.write_nodata(no_data, inplace=True)

        # reproject and create new xarray "raster"
        self.raster = self.datacube[data_vars].rio.reproject(epsg, nodata=no_data)

    def open_file(self, file):
        """
        Legacy (GRS <= v1.5): open a netCDF product and set its CRS and
        geotransform from the 'crs' variable (wkt and i2m attributes).
        """
        product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                                  decode_coords='all')

        # set CRS
        epsg = rxr.crs.CRS.from_wkt(product.crs.wkt).to_epsg()
        self.epsg = epsg
        product.rio.write_crs(epsg, inplace=True)

        # set geotransform
        i2m = np.array((product.crs.i2m.split(','))).astype(float)
        gt = Affine(i2m[0], i2m[1], i2m[4], i2m[2], i2m[3], i2m[5])
        product.rio.write_transform(gt, inplace=True)

        # TODO add coords x, y (missing for GRS <= v1.5): ``assign_coords`` was never implemented
        return product
