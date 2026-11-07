#!/usr/bin/env python
# coding: utf-8
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # activate the rio accessor

# import cartopy
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import locale

locale.setlocale(locale.LC_ALL, 'en_US.utf8')
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')


import pyproj as ppj

from datetime import datetime
from dateutil import rrule
from dateutil.relativedelta import relativedelta

import grstbx

u = grstbx.utils
opj = os.path.join

grstbx.__version__

ppj.datadir.get_data_dir()
ppj.datadir.set_data_dir('/work/scratch/harmelt/envs/grstbx/share/proj')

idir = '/datalake/watcal'
odir = '/datalake/watcal/datacube'

tile = '31TGM'
product = 'S2-L2GRS'
pattern = '*15.nc'
# central coordinate
lon, lat = 6.55, 46.4
lon, lat = 6.56, 46.37
ID = 'leman'
# size in meter of the rectangle of interest
width, height = 35000, 21000
#width, height = 1000, 1000

# dates
start_date = datetime(2016, 1, 1)
end_date = datetime(2023, 1, 1)

for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

    start_date_, end_date_ = dt.strftime('%Y-%m-%d'), (dt + relativedelta(day=31)).strftime('%Y-%m-%d')
    date_month = dt.strftime('%Y-%m')
    ofile = opj(odir, tile + '_' + date_month + '_' + ID + '.nc')

    if os.path.exists(ofile):
        continue
    print(ofile)

    select = grstbx.SelectFiles()

    # if you need to change the root path
    select.root = idir
    select.list_tile(tile=tile, product=product, pattern=pattern)
    files = select.list_file_path((start_date_, end_date_))

    ### Drop dates with too high cloud coverage

    select.file_list = select.file_list[select.file_list.cloud_coverage < 60]
    select.files = select.file_list.abspath.values
    print(select.files)
    if len(select.files)==0:
        continue
    print(select.files)
    ## Load and subset image series

    ust = grstbx.utils.SpatioTemp()
    box = ust.wktbox(lon, lat, width=width, height=height, ellps='WGS84')
    bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
    # reproject lon lat in xy coordinates
    # bbox = bbox.to_crs(epsg=32631)

    # generate datacube
    dc = grstbx.L2grs(select.files)
    dc.load(subset=bbox, reshape=False, reproject=False)

    # filter out empty rasters
    print('filter out empty rasters')
    threshold = 0.002
    Npix = dc.datacube.Rrs_B2.count(dim=['x', 'y'])
    times = Npix[Npix / Npix.max().values > threshold].time.values
    if len(times) == 0:
        continue
    dc.datacube = dc.datacube.sel(time=times)

    # mask remaining rasters
    print('mask remaining rasters')
    masking_ = grstbx.Masking(dc.datacube)
    mask = masking_.get_mask(high_nir=True) | masking_.get_mask(hicld=True) | (dc.datacube.Rrs_B3 < 0.0002) | (
                dc.datacube.Rrs_B3 > 0.12)

    # compute number of valid pixels per date
    mask_ts = (~mask).sum(['x', 'y']).compute()

    valid_pix = mask_ts / mask_ts.max()

    times = valid_pix[valid_pix > threshold].time.values
    if len(times) == 0:
        continue
    dc.datacube = dc.datacube.sel(time=times)
    mask = mask.sel(time=times)
    print('reproject full raster')
    dc.datacube = dc.datacube.where(mask == 0)
    dc.reproject_data_vars(data_vars=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                      'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                      'Rrs_B8A', 'SZA', 'AZI', 'VZA', 'shade', 'BRDFg'])
    dc.reshape_raster(data_vars=['SZA', 'AZI', 'VZA', 'shade', 'BRDFg'])
    ds = dc.raster
    # remove _FillValue attributes allocated during reprojection to avoid conflict with netcdf encoding
    variables = list(ds.keys())
    for variable in variables:
        if ds[variable].attrs.__contains__('_FillValue'):
            del ds[variable].attrs['_FillValue']
    print('exprot into netcdf')
    encoding = {'Rrs': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .3,
                        '_FillValue': -32768, 'zlib': True, 'complevel': 5},
                'SZA': {'dtype': 'int16', 'scale_factor': 0.001,
                        '_FillValue': -32768, 'zlib': True, 'complevel': 5},
                'VZA': {'dtype': 'int16', 'scale_factor': 0.001,
                        '_FillValue': -32768, 'zlib': True, 'complevel': 5},
                'AZI': {'dtype': 'int16', 'scale_factor': 0.001,
                        '_FillValue': -32768, 'zlib': True, 'complevel': 5},
                'shade': {'dtype': 'int16', 'scale_factor': 0.001,
                          '_FillValue': -32768, 'zlib': True, 'complevel': 5},
                'BRDFg': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .3,
                          '_FillValue': -32768, 'zlib': True, 'complevel': 5}}

    ds.to_netcdf(ofile, encoding=encoding)
    del ds, dc
