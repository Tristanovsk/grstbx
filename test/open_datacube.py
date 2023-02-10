#!/usr/bin/env python
# coding: utf-8


import glob
import os
import numpy as np, pandas as pd, xarray as xr
import geopandas as gpd
import dask
import rioxarray  # activate the rio accessor
import matplotlib as mpl
mpl.use('TkAgg')
#import cartopy
import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.utf8')
import matplotlib.pyplot as plt

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

tile='30TXR'
ID='gironde'
odir ='/sat_data/satellite/sentinel2/datacube/'
odir ='/home/harmel/Dropbox/satellite/S2/cnes/datacube/'

start_date='2022-07-01'
stop_date='2022-12-31'
basename= tile+'_'+start_date+'_'+stop_date+'_'+ID
basename='31TGM_2022-01-01_2022-06-30_SHL2'
ofile = opj(odir,basename+'.nc')

# Open and load your datacube:
# from netcdf format and interpret coordinate system
raster = xr.open_dataset(ofile,decode_coords='all')
dc =grstbx.l2grs()
dc.raster=raster
#dc.reshape_raster()

coarsening = 1
aspect=len(dc.raster.x)/len(dc.raster.y)*1.2

fig = dc.raster.Rrs.isel(wl=2)[:,::coarsening, ::coarsening].plot(col='time', col_wrap=4,robust=True,cmap=cc.cm.CET_D13,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.savefig('test/fig/'+basename+'_B3.png',dpi=300)


# In[17]:


#coarsening = 2
fig=raster.BRDFg[:,::coarsening, ::coarsening].plot(col='time', col_wrap=4,vmax=0.01,robust=True,cmap=cc.cm.gray,aspect=aspect)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.savefig('test/fig/'+basename+'_sunglint.png',dpi=300)


# In[18]:


bands=[4,2,1]
bands=[3,2,1]
coarsening = 1
brightness_factor = 5
gamma=2
fig = (dc.raster.Rrs.isel(wl=bands)[:,:,::coarsening, ::coarsening]**(1/gamma)*brightness_factor).plot.imshow(col='time', col_wrap=4,robust=True,aspect=aspect/1.2)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.savefig('test/fig/'+basename+'_RGB.png',dpi=300)

