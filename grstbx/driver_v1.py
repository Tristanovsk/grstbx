import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr  # activate the rio accessor
import geopandas as gpd

# just to make sure that with the good format for month:
import locale

locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pyproj as ppj
from affine import Affine
from datetime import datetime as dt

from .masking import masking


class l2grs():
    def __init__(self, files=None):
        self.files = files
        self.no_product = False
        self.raster=None

    def load(self, reproject=False, subset=None, reshape=True):
        '''
        :param reproject: if True reproject in geodetic coordinates, keep native projection otherwise 
        :param subset: None for no subset or [minx, miny, maxx, maxy] in the appropriate crs
        :return:
        '''
        self.get_datacube(reproject=reproject, subset=subset)
        if self.no_product == True:
            print('no product available for your settings')
            return
        if reshape:
            self.reshape_raster(from_datacube=True)

    # @staticmethod
    def assign_coords(self, product, reproject=False, epsg_in=4326, epsg_out=3857, res=None):

        i2m = np.array((product.crs.i2m.split(','))).astype(float)
        if res == None:
            res = i2m[0]  # float(product.metadata.attrs['Processing_Graph:node_1:parameters:resolution'])
        x_, y_ = product.x, product.y
        gt = product.rio.transform()
        x0, y0 = gt * (y_ + 0.5, x_ + 0.5)
        product['x'] = x0[:, 0].values
        product['y'] = y0[0, :].values

        if reproject:
            return product.rio.reproject(epsg_out)
        return product

    @staticmethod
    def assign_coords_pyproj(product, epsg_in=4326, epsg_out=3857):

        transformer = ppj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
        nx, ny = product.lon.shape
        # TODO try to get lat lon min max differently, here it is inefficient 
        x, y = transformer.transform([product.lon.min(), product.lon.max()],
                                     [product.lat.min(), product.lat.max()])
        product['x'] = np.linspace(x[0], x[1], nx)
        product['y'] = np.linspace(y[0], y[1], ny)

        return product

    @staticmethod
    def add_time_dim(xda):
        time = [dt.strptime(xda.attrs['start_date'], '%d-%b-%Y %H:%M:%S.%f')]
        xda = xda.assign_coords(time=time)
        # xda = xda.expand_dims('time')
        return xda

    def subset_xy(self, ds, bbox):

        bbox = bbox.to_crs(epsg=self.epsg)
        minx, miny, maxx, maxy = bbox.bounds.values[0]
        return ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))

    def get_datacube(self, subset=None, reproject=False, epsg_out=3857):
        # product = xr.open_mfdataset(self.files, chunks={'x': 512, 'y': 512},
        #                     decode_coords='all',combine='nested',
        #                     concat_dim="time",preprocess = self.add_time_dim,
        #                     parallel=True)
        products = []
        for file in self.files:

            product = self.open_file(file)  # , reproject=reproject, epsg_in=self.epsg,epsg_out=epsg_out)

            if subset is not None:
                product = self.subset_xy(product, subset)

            if reproject:
                product = product.rio.reproject(epsg_out)
                self.epsg = product.rio.crs.to_epsg()

            # add time dimension
            product = self.add_time_dim(product)

            # automatically remove product when no data available
            masking_ = masking(product)
            nodata = masking_.get_mask(nodata=False)  # .compute()
            if nodata.sum().compute() == 0:
                continue

            # ---------------------------------------------------
            # fix merging when MAJA flags are not always present
            # get names of variables
            varnames = pd.DataFrame(product.data_vars)
            vars_to_remove = varnames[0][varnames[0].str.contains("mask")]
            product = product.drop_vars(vars_to_remove)
            try:
                product = product.drop_vars('aot_maja')
            except:
                pass

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

        # ---------------------------------------------------
        # fix for nodata value of BRDFg
        # TODO need to check in grs (to be removed when solved)
        self.datacube['BRDFg'] =self.datacube['BRDFg'].where(self.datacube[
                                                                   'BRDFg'] > -99)
        # ---------------------------------------------------



    def reshape_raster(self, bands=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                    'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                    'Rrs_B8A'],
                         data_vars=[ 'SZA', 'AZI', 'VZA', 'shade', 'BRDFg'],
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
            #if self.datacube[var].dtype float32:
            self.datacube[var].rio.write_nodata(no_data,inplace=True)
            #elif isinstance(self.datacube[var], int):
            #    self.datacube[var].rio.write_nodata(-999,inplace=True)

        # reproject and create new xarray "raster"
        self.raster = self.datacube[data_vars].rio.reproject(epsg, nodata=no_data)


    def open_file(self, file):
        product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                                  decode_coords='all')  # ,engine='netcdf4')

        # remove lat lon raster to avoid conflict with PROJ, GDAL, RIO
        product = product.drop(['lat', 'lon'])

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
#
# import glob
# opj = os.path.join
# satdir = '/sat_data/satellite/sentinel2/L2A/GRS/31TGM'
#
# image = 'S2A_MSIl2grs_20210906T103021_N0301_R108_T31TGM_20210906T141939_cc004_v14.nc'
# files = glob.glob(opj(satdir, image))
#
# dc = l2grs(files)
# dc.load()
#
# Rrs.load()
# # merge to keep flags
# ds = xr.merge([Rrs,p_.flags])
# masking_ = masking(ds)
# mask_ = masking_.get_mask(negative=True,nodata=False)
