import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr  # activate the rio accessor
import geopandas as gpd

# just to make sure that with the good format for month:
import locale

# locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pyproj as ppj
from affine import Affine
from datetime import datetime as dt

from .masking import Masking

opj = os.path.join


class L2grs():

    def __init__(self, files):
        self.files = files
        self.xchunk = 5490
        self.ychunk = 5490
        self.wlchunk = -1

    def load_l2a_image(self,
                       l2a_path):

        if l2a_path.split('.')[-1] in 'zarr':
            return xr.open_zarr(l2a_path,decode_coords='all'), None

        basename = os.path.basename(l2a_path)
        main_file = opj(l2a_path, basename + '.nc')
        ancillary_file = opj(l2a_path, basename + '_anc.nc')

        raster = xr.open_dataset(main_file, decode_coords='all',
                                 chunks={'wl': self.wlchunk, 'x': self.xchunk, 'y': self.ychunk})
        ancillary = xr.open_dataset(ancillary_file, decode_coords='all')

        if raster.attrs['metadata_profile'] != 'beam':
            return raster, ancillary

        # reshape into datacube:
        wls = raster.wl

        if raster.dims.__contains__('wl'):
            raster = raster.drop_dims('wl')

        Rrs_vars = []
        Rrs_g_vars = []
        for wl in wls:
            Rrs_vars.append('Rrs_{:d}'.format(wl))
            # Rrs_g_vars.append('Rrs_g_{:d}'.format(wl))

        Rrs = raster[Rrs_vars].to_array(dim='wl', name='Rrs').chunk(
            {'wl': self.wlchunk, 'x': self.xchunk, 'y': self.ychunk})
        Rrs = Rrs.assign_coords({'wl': wls})
        raster = raster.drop_vars(Rrs_vars)
        # Rrs_g = raster[Rrs_g_vars].to_array(dim='wl', name='Rrs_g').chunk(
        #    {'wl': self.wlchunk, 'x': self.xchunk, 'y': self.ychunk})
        # Rrs_g = Rrs_g.assign_coords({'wl': wls})
        raster = raster.drop_vars(Rrs_g_vars)
        # return xr.merge([raster, Rrs, Rrs_g]), ancillary
        return xr.merge([raster, Rrs]), ancillary

    def load_l2b_image(self, l2b_path):
        return xr.open_dataset(l2b_path, decode_coords='all', chunks={'x': self.xchunk, 'y': self.ychunk})

    def subset_xy(self, ds, bbox):

        bbox = bbox.to_crs(epsg=ds.rio.crs.to_epsg())
        minx, miny, maxx, maxy = bbox.bounds.values[0]
        return ds.sel(x=slice(minx, maxx), y=slice(maxy, miny)).load()

    def get_l2a_datacube(self,
                         subset=None,
                         reproject=False,
                         nodata_thresh=0.5,
                         epsg_out=3857):
        # product = xr.open_mfdataset(self.files, chunks={'x': 512, 'y': 512},
        #                     decode_coords='all',combine='nested',
        #                     concat_dim="time",preprocess = self.add_time_dim,
        #                     parallel=True)
        FLAG_NAME = 'flags'
        products = []
        for file in self.files:

            product, anc = self.load_l2a_image(file)  # , reproject=reproject, epsg_in=self.epsg,epsg_out=epsg_out)

            # add mean solar angles:
            for attribute in ['mean_solar_azimuth', 'mean_solar_zenith_angle']:
                product[attribute] = product.attrs[attribute]

            if subset is not None:
                product = self.subset_xy(product, subset)

            if reproject:
                product = product.rio.reproject(epsg_out)
                self.epsg = product.rio.crs.to_epsg()

            # get flag statistics and mask
            flag_stats = self.get_flag_stats(product[FLAG_NAME].expand_dims('time'))
            if flag_stats.flag_nodata.values > nodata_thresh:
                continue

            # product = xr.merge([product,flag_stats])

            products.append(product)

        if len(products) == 0:
            self.no_product = True
            return

        product = xr.concat(products, dim='time')  # .sortby('time')
        # keep only one date for dem
        product['dem'] = product.dem.isel(time=0)

        # add flags statistics:
        product = xr.merge([product, self.get_flag_stats(product.flags)])

        self.datacube = product
        self.datacube.attrs['start_date'] = str(product.time[0].values)
        self.datacube.attrs['stop_date'] = str(product.time[-1].values)
        self.pixnum = len(self.datacube.x) * len(self.datacube.y)

    def get_l2b_datacube(self,
                         subset=None,
                         reproject=False,
                         epsg_out=3857,
                         var='Chla_OC2nasa',
                         var_novalid='central_wavelength'):
        # product = xr.open_mfdataset(self.files, chunks={'x': 512, 'y': 512},
        #                     decode_coords='all',combine='nested',
        #                     concat_dim="time",preprocess = self.add_time_dim,
        #                     parallel=True)
        products = []
        for file in self.files:

            product = self.load_l2b_image(file)  # , reproject=reproject, epsg_in=self.epsg,epsg_out=epsg_out)

            if var_novalid in product:
                product = product.drop_vars(var_novalid)

            if subset is not None:
                product = self.subset_xy(product, subset)

            if reproject:
                product = product.rio.reproject(epsg_out)
                self.epsg = product.rio.crs.to_epsg()

            # check valid pixels:
            width, height = len(product.x.values), len(product.y.values)
            Npix_tot = width * height
            Npix_valid = product[var].count().compute()
            product['valid_pix_prop'] = Npix_valid / Npix_tot
            if product['valid_pix_prop'] == 0:
                continue
            product['valid_pix_prop'].attrs['description'] = 'Proportion of valid pixels for ' + var \
                                                             + ' within the image raster'
            # # automatically remove product when no data available
            # masking_ = masking(product)
            # nodata = masking_.get_mask(nodata=False)  # .compute()
            # if nodata.sum().compute() == 0:
            #     continue
            #
            # # ---------------------------------------------------
            # # fix merging when MAJA flags are not always present
            # # get names of variables
            # varnames = pd.DataFrame(product.data_vars)
            # vars_to_remove = varnames[0][varnames[0].str.contains("mask")]
            # product = product.drop_vars(vars_to_remove)
            # try:
            #     product = product.drop_vars('aot_maja')
            # except:
            #     pass

            # ----------------------------------------------------

            products.append(product)

        if len(products) == 0:
            self.no_product = True
            return

        product = xr.concat(products, dim='time').sortby('time')
        self.datacube = product
        self.datacube.attrs['start_date'] = str(product.time[0].values)
        self.datacube.attrs['stop_date'] = str(product.time[-1].values)
        self.pixnum = len(self.datacube.x) * len(self.datacube.y)

    @staticmethod
    def get_flag_stats(raster):
        '''
        Compute statistics for each flag and each time / exprot into xarray

        :param raster: multitemporal L2a xarray.Dataset
        :return:
        '''

        flag_stats = {}
        first = True
        for itime, date in enumerate(raster.time):
            raster_ = raster.isel(time=itime)
            flag_value = 1
            for ii, flag_name in enumerate(raster_.flag_names):
                if flag_name != 'None':
                    flag = ((raster_ & flag_value) != 0)
                    flag_stat = float(flag.sum() / flag.count())
                    if first:
                        flag_stats['flag_' + flag_name] = {"dims": "time", "data": []}
                    flag_stats['flag_' + flag_name]['data'].append(flag_stat)
                flag_value = flag_value << 1
            first = False

        return xr.Dataset.from_dict(flag_stats).assign_coords({'time': raster.time.values})

    def reshape_raster(self, bands=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                    'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                    'Rrs_B8A'],
                       data_vars=['SZA', 'AZI', 'VZA', 'shade', 'BRDFg'],
                       from_datacube=False
                       ):
        if from_datacube:
            p_ = self.datacube
        else:
            p_ = self.raster
        wl = []
        for name, band in p_[bands].items():
            wl.append(band.attrs['wavelength'])

        Rrs = p_[bands]  # .squeeze()
        Rrs = Rrs.to_array(dim='wl', name='Rrs').assign_coords(wl=wl).chunk({'wl': 1})

        # merge to keep flags
        self.raster = xr.merge([Rrs, p_[data_vars]]).chunk({'time': 1})
        return

    def reproject_data_vars(self, epsg=3857,
                            data_vars=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                       'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                       'Rrs_B8A' 'SZA', 'AZI', 'VZA', 'shade', 'BRDFg'],
                            no_data=np.nan
                            ):

        # set no_data value before reprojection
        for var in data_vars:
            # TODO handle differnt nodata value depending on dtype (int, float, str...)
            # if self.datacube[var].dtype float32:
            self.datacube[var].rio.write_nodata(no_data, inplace=True)
            # elif isinstance(self.datacube[var], int):
            #    self.datacube[var].rio.write_nodata(-999,inplace=True)

        # reproject and create new xarray "raster"
        self.raster = self.datacube[data_vars].rio.reproject(epsg, nodata=no_data)

    def open_file(self, file):
        product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                                  decode_coords='all')  # ,engine='netcdf4')

        # remove lat lon raster to avoid conflict with PROJ, GDAL, RIO
        # product = product.drop(['lat', 'lon'])

        # set CRS
        epsg = rxr.crs.CRS.from_wkt(product.crs.wkt).to_epsg()
        # TODO check if that dat is necessary (normally rio handles it)
        self.epsg = epsg
        product.rio.write_crs(epsg, inplace=True)

        # set geotransform
        # i2m = product.crs.i2m
        i2m = np.array((product.crs.i2m.split(','))).astype(float)
        gt = Affine(i2m[0], i2m[1], i2m[4], i2m[2], i2m[3], i2m[5])
        product.rio.write_transform(gt, inplace=True)

        # add coords x, y (missing for GRS <= v1.5)
        return self.assign_coords(product)
