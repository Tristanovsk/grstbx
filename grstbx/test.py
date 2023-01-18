import os
import glob
import geopandas as gpd
import numpy as np
import pandas as pd

import grstbx
from datetime import datetime as dt
import xarray as xr
import rioxarray as rxr  # activate the rio accessor
import matplotlib.pyplot as plt
# just to make sure that with the good format for month:
import locale
import pyproj as ppj
from affine import Affine
locale.setlocale(locale.LC_ALL, 'en_US.utf8')
opj = os.path.join


#######################
# mosaic
root='/media/harmel/vol1/Dropbox/satellite/S2/cnes'
file1=glob.glob(opj(root,'37PGN','2022','12','01','*.nc'))[0]
file2=glob.glob(opj(root,'37PHN','2022','12','01','*.nc'))[0]
tbx = grstbx.l2grs()
img1 = tbx.open_file(file1)
img2 = tbx.open_file(file2)

############################################################
tile='31TEJ'
select =grstbx.select_files()
# if you need to change the root path
select.root='/home/harmel/Dropbox/satellite/S2'
select.root='/media/harmel/TOSHIBA EXT/data/satellite'
select.list_tile(tile=tile,product='S2_L2GRS',pattern='*v14.nc')

files = select.list_file_path(('2022-01-12','2022-01-31'))
# central coordinate
lon, lat = 3.61, 43.4
# size in meter of the rectangle of interest
width, height = 10000, 10000

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
bbox = bbox.to_crs(epsg=32631)

# generate datacube
dc = grstbx.l2grs(files)
dc.load(subset=bbox)#,reproject=True)

B2 = dc.datacube.Rrs_B2.rio.reproject(3857)
B2.plot(col='time',robust=True,vmin=0)

############################################################
# test processing a single image


satdir = '/sat_data/satellite/sentinel2/L2A/GRS/31TGM'
epsg_out=3857
image = 'S2*_v14.nc'
files = glob.glob(opj(satdir, image))
product = xr.open_dataset(files[0], chunks={'x': 512, 'y': 512},
                                      decode_coords='all',engine='netcdf4')

# remove lat lon raster to avoid conflict with PROJ, GDAL, RIO
product = product.drop(['lat', 'lon'])

# set CRS
epsg = rxr.crs.CRS.from_wkt(product.crs.wkt).to_epsg()
product.rio.write_crs(epsg, inplace=True)

# set geotransform
i2m = product.crs.i2m
i2m = np.array((product.crs.i2m.split(','))).astype(float)
gt = Affine(i2m[0], i2m[1], i2m[4], i2m[2], i2m[3],i2m[5])
product.rio.write_transform(gt, inplace=True)

# add coords x, y (missing for GRS <= v1.5)
x_, y_ = product.x, product.y
gt = product.rio.transform()
x0, y0 = gt * (y_ + 0.5, x_ + 0.5)
product['x'] = x0[:, 0].values
product['y'] = y0[0, :].values

# reproject:
product_proj = product.rio.reproject(epsg_out,nodata=np.nan)

# product = xr.open_dataset(files[0], chunks={'x': 512, 'y': 512}, decode_coords="all",engine='netcdf4')
# product = product.drop(['lat','lon'])
# i2m = product.crs.i2m
# wkt = product.crs.wkt
# #set geotransform
# i2m = np.array((product.crs.i2m.split(','))).astype(float)
# gt = Affine(i2m[0], i2m[1], i2m[4], i2m[2], i2m[3],i2m[5])
# product.rio.write_transform(gt, inplace=True)
# product.rio.write_crs(32631,inplace=True)
# gt = product.rio.transform()
# x_,y_ = product.x,product.y
# nx,ny=len(x_),len(y_)
# res=20
# x0,y0 = gt*(y_+0.5,x_+0.5)
# product['x']= x0[:,0].values
# product['y']= y0[0,:].values
#
# x1, y1 = gt* (nx, ny)
# product['x'] = np.arange(x0 + res / 2, x1 - 1, res)
# product['y'] = np.arange(y0 - res / 2, y1 + 1, -res)
#
# product.crs.wkt.split('\n')[-1]
ust = grstbx.utils.spatiotemp()
box = ust.wktbox(6.6, 46.45, width=1000, height=1000, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
print(bbox.bounds)
bbox = bbox.to_crs(epsg=32631)
print(bbox.bounds)
bbox = bbox.to_crs(epsg=3857)
print(bbox.bounds)

dc = grstbx.l2grs(files,)
dc.load(subset=bbox)
#bbox = bbox.to_crs(epsg=3857)
print(bbox.bounds)
dc2 = grstbx.l2grs(files,)
dc2.load(reproject=True,subset=bbox)
dc.Rrs.Rrs.rio.reproject('EPSG:3857')
dc.Rrs.Rrs.plot(col='wl',row='time',vmin=0)
dc2.Rrs.Rrs.plot(col='wl',row='time',vmin=0)

#test geotransform
gt = (699960.0, 20.0, 0.0, 5200020.0, 0.0, -20.0)
xpix=np.arange( 5490)
ypix=np.arange( 5490)
x = gt[0]+xpix * gt[1]+ypix * gt[2]
y = gt[3]+xpix * gt[4]+ypix * gt[5]

geotrans = ppj.Transformer.from_crs( 32631,4326, always_xy=True)#,authority="EPSG")
#x, y = product['x'].values, product['y'].values
geotrans.transform(x, y)
geotrans = ppj.Transformer.from_crs(32631, 3857, always_xy=True)#,authority="EPSG")
#x, y = product['x'].values, product['y'].values
x,y=geotrans.transform(x, y)

geotrans = ppj.Transformer.from_crs( 3857,4326, always_xy=True)#,authority="EPSG")
#x, y = product['x'].values, product['y'].values
geotrans.transform(x, y)
def add_time_dim(xda):
    time=[dt.strptime(xda.attrs['start_date'], '%d-%b-%Y %H:%M:%S.%f')]
    xda = xda.assign_coords(time=time)
    #xda = xda.expand_dims('time')
    return xda

product = xr.open_mfdataset(files[0:1], chunks={'x': 512, 'y': 512},
                            decode_coords='all',combine='nested',
                            concat_dim="time",preprocess = add_time_dim,
                            parallel=True)

