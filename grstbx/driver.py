import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # activate the rio accessor
import geopandas as gpd

# just to make sure that with the good format for month:
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pyproj as ppj
from affine import Affine
from datetime import datetime as dt


class l2grs():
    def __init__(self, files):
        self.files = files

    def load(self, reproject=False,subset=None):
        '''
        :param reproject: if True reproject in geodetic coordinates, keep native projection otherwise 
        :param subset: None for no subset or [minx, miny, maxx, maxy] in the appropriate crs
        :return:
        '''
        self.get_datacube(reproject=reproject,subset=subset)
        self.reshape_datacube()

    #@staticmethod
    def assign_coords(self, product, reproject=False, epsg_in=4326,epsg_out=3857, res=None):
        if res == None:
            res = float(product.metadata.attrs['Processing_Graph:node_1:parameters:resolution'])
        i2m = np.array((product.crs.i2m.split(','))).astype(float)
        geotransform = (i2m[4], i2m[0], i2m[1], i2m[5], i2m[2], i2m[3])
        fwd = Affine.from_gdal(*geotransform)
        nx, ny = len(product.x), len(product.y)
        x0, y0 = fwd * (0, 0)
        x1, y1 = fwd * (nx, ny)
        wkt_poly = 'POLYGON (( %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f ))' % (
        x0, y0, x0, y1, x1, y1, x1, y0,x0,y0)
        bbox = gpd.GeoSeries.from_wkt([wkt_poly]).set_crs(epsg=epsg_in)
        if reproject:            
            bbox = bbox.to_crs(epsg=epsg_out)
            x0,y0,x1,y1 = bbox.bounds.values[0]
            product['x'] = np.linspace(x0, x1, nx)
            product['y'] = np.linspace(y1, y0, ny)

        else:
            product['x'] = np.arange(x0 + res / 2, x1 - 1, res)
            product['y'] = np.arange(y0 - res / 2, y1 + 1, -res)
        self.bbox=bbox
        return product
    
    @staticmethod
    def assign_coords_pyproj(product, epsg_in=4326,epsg_out=3857):
     
        transformer = ppj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
        nx,ny = product.lon.shape
        # TODO try to get lat lon min max differently, here it is inefficient 
        x, y = transformer.transform([product.lon.min(), product.lon.max()],
                                     [product.lat.min(), product.lat.max()] )
        product['x'] = np.linspace(x[0], x[1], nx)
        product['y'] = np.linspace(y[0], y[1], ny)

        return product
    
    @staticmethod
    def add_time_dim(xda):
        time = [dt.strptime(xda.attrs['start_date'], '%d-%b-%Y %H:%M:%S.%f')]
        xda = xda.assign_coords(time=time)
        # xda = xda.expand_dims('time')
        return xda

    @staticmethod
    def subset_xy(ds, minx, miny, maxx, maxy):
        return ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))

    def get_datacube(self, subset=None,reproject=False):
        # product = xr.open_mfdataset(self.files, chunks={'x': 512, 'y': 512},
        #                     decode_coords='all',combine='nested',
        #                     concat_dim="time",preprocess = self.add_time_dim,
        #                     parallel=True)
        products = []
        for file in self.files:
            product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                                      decode_coords='all')
            product = self.add_time_dim(product)

            epsg = ppj.CRS.from_wkt(product.crs.wkt).to_epsg()
            # product = product.metpy.assign_crs(crs.to_cf()).metpy.assign_y_x()            
            #product = self.assign_coords_pyproj(product,epsg_out=32631)
            product = self.assign_coords(product,reproject=reproject, epsg_in=epsg)
            #product.rio.write_crs(epsg, inplace=True)
            if subset is not None:
                product = self.subset_xy(product, *subset)
            products.append(product)
        product = xr.concat(products,dim='time').sortby('time')
        self.datacube = product#.rio.write_coordinate_system()

    def reshape_datacube(self, bands=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                      'Rrs_B8A'],
                         variables=['flags', 'SZA', 'shade']):
        p_ = self.datacube
        wl = []
        for name, band in p_[bands].items():
            wl.append(band.attrs['wavelength'])

        Rrs = p_[bands]  # .squeeze()
        Rrs = Rrs.to_array(dim='wl', name='Rrs').assign_coords(wl=wl).chunk({'wl': 1})

        # merge to keep flags
        self.Rrs = xr.merge([Rrs, p_[variables]]).chunk({'time': 1})
        return
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
