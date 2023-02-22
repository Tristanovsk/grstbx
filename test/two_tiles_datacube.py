
import glob
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # activate the rio accessor

#import cartopy.feature as cfeature
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import colorcet as cc

import pyproj as ppj
import rasterio
from affine import Affine
from shapely.geometry import box,Point, mapping
from shapely.ops import transform


import grstbx


u = grstbx.utils
opj = os.path.join

grstbx.__version__


tile='37PGN'
idir='/media/harmel/vol1/Dropbox/satellite/S2'
select =grstbx.select_files()

# if you need to change the root path
select.root=idir
select.list_tile(tile=tile,product='cnes',pattern='*15.nc')
tile1 = select.list_file_path(('2022-12','2022-12-31'))


tile = '37PHN'
select.list_tile(tile=tile,product='cnes',pattern='*15.nc')
tile2 = select.list_file_path(('2022-12','2022-12-31'))


# central coordinate
lon, lat = 41.78, 11.18
# size in meter of the rectangle of interest
width, height = 15000, 15000
aspect=width/height

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
#bbox = bbox.to_crs(epsg=32631)

# generate datacube
dc1 = grstbx.l2grs(tile1)
dc1.load(subset=bbox,reproject=True, reshape=False)


# generate datacube
dc2 = grstbx.l2grs(tile2)
dc2.load(subset=bbox,reproject=True)
from rioxarray.merge import merge_datasets
dc1.Rrs.Rrs.rio.write_nodata(np.nan,inplace=True)
dc2.Rrs.Rrs.rio.write_nodata(np.nan,inplace=True)
dc = merge_datasets([dc1.Rrs.isel(wl=2),dc2.Rrs.isel(wl=2)],method='min')
coarsening = 2
fig = dc1.Rrs.Rrs.isel(wl=2)[:,::coarsening, ::coarsening].plot.imshow(col='time', col_wrap=4,robust=True,aspect=aspect)
fig = dc2.Rrs.Rrs.isel(wl=2)[:,::coarsening, ::coarsening].plot.imshow(col='time', col_wrap=4,robust=True,aspect=aspect)
fig = dc.Rrs.drop_isel(time=4)[:,::coarsening, ::coarsening].plot.imshow(col='time', col_wrap=3,robust=True,cmap=cc.cm.CET_D13,aspect=aspect)

# remove bad dates
Rrs = dc.Rrs.drop_isel(time=4)
# plot stats
Rrs_mean = Rrs.mean('time')
Rrs_median = Rrs.median('time')
Rrs_std = Rrs.std('time')

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(15, 4))
Rrs_mean.plot.imshow(ax=axs[0],robust=True,cmap=cc.cm.CET_D13)
Rrs_median.plot.imshow(ax=axs[1],robust=True,cmap=cc.cm.CET_D13)
Rrs_std.plot.imshow(ax=axs[2],robust=True,cmap=cc.cm.bkr)
axs[0].set_title("Mean")
axs[1].set_title("Median")
axs[2].set_title("StD")

for ax in axs:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.savefig('illustration/test_abhe_stats_dec2022.png',dpi=300)

bands=[4,2,1]
#bands=[3,2,1]



brightness_factor = 5
gamma=2
fig = (dc.Rrs.isel(wl=bands)[:,:,::coarsening, ::coarsening]**(1/gamma)*brightness_factor).plot.imshow(col='time', col_wrap=4,robust=True,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig

dc.to_netcdf('test/test_mosaic.nc')