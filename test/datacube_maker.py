#!/usr/bin/env python
# coding: utf-8


import glob
import os
import numpy as np, pandas as pd, xarray as xr
import geopandas as gpd
import dask
import rioxarray  # activate the rio accessor

#import cartopy
import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.utf8')
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')
import colorcet as cc

import pyproj as ppj
import rasterio
from affine import Affine
from shapely.geometry import box,Point, mapping
from shapely.ops import transform


import grstbx
from grstbx import visual

u = grstbx.utils
opj = os.path.join

grstbx.__version__

# ## Set PROJ path if necessary

ppj.datadir.get_data_dir()
#ppj.datadir.set_data_dir('/work/scratch/harmelt/envs/grstbx/share/proj')


# ## Set the images you want to play with

tile='37PGN'
ID='abhe'
odir ='/sat_data/satellite/sentinel2/datacube/'
start_date='2022-11-01'
stop_date='2022-12-31'
ofile = opj(odir,tile+'_'+start_date+'_'+stop_date+'_'+ID+'.nc')

idir='/media/harmel/vol1/Dropbox/satellite/S2'
idir='/datalake/watcal'
select =grstbx.select_files()

# if you need to change the root path
select.root=idir
select.list_tile(tile=tile,product='S2-L2GRS',pattern='*15.nc')
files = select.list_file_path((start_date,stop_date))
files

select.file_list


# ## Load and subset image series

# central coordinate
lon, lat = 41.77, 11.18 
# size in meter of the rectangle of interest
width, height = 13000, 15000
aspect=width/height

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
#bbox = bbox.to_crs(epsg=32631)

# generate datacube
dc = grstbx.l2grs(files)
dc.load(subset=bbox,reshape=False, reproject=False)


# ## Check data/metadata

masking_ = grstbx.masking(dc.datacube)
dc.datacube


# ## Mask datacube
# Mask pixels from chosen flags and remove empty dates

mask = (masking_.get_mask(high_nir=True) | masking_.get_mask(hicld=True)) | (dc.datacube.Rrs_B3 < 0.0002)
mask.compute()

# compute number of valid pixels per date
mask_ts =  (~mask).sum(['x','y']).compute()


valid_pix = mask_ts / mask_ts.max()
valid_pix.plot()


# Drop dates with valid pixel below a threshold, for example keep date when valid pixels are at least 60%

threshold = 0.6
dc.datacube = dc.datacube.where(valid_pix>threshold,drop=True)



# Mask raster for each remaining date

dc.datacube = dc.datacube.where(mask==0,drop=True)


# reproject desired variables into desired epsg (by default 3857 for basemap projection)

dc.reproject_data_vars()


# reshape your raster to add the wavelength diemension to the requested Rrs

dc.reshape_raster()


# ## **Fast checking of the RGB images**

# You are done with your datacube:
# save it into netcdf format
dc.raster.to_netcdf(ofile)

coarsening = 3
fig = dc.raster.Rrs.isel(wl=2)[:,::coarsening, ::coarsening].plot(col='time', col_wrap=4,robust=True,cmap=cc.cm.CET_D13,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[17]:


coarsening = 2
fig=dc.raster.BRDFg[:,::coarsening, ::coarsening].plot(col='time', col_wrap=4,vmax=0.01,robust=True,cmap=cc.cm.gray,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[18]:


bands=[4,2,1]
#bands=[3,2,1]
coarsening = 2
brightness_factor = 5
gamma=2
fig = (dc.raster.Rrs.isel(wl=bands)[:,:,::coarsening, ::coarsening]**(1/gamma)*brightness_factor).plot.imshow(col='time', col_wrap=4,robust=True,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig

