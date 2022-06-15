import os
import glob

from datetime import datetime as dt
import xarray as xr

opj = os.path.join
import geopandas as gpd
from shapely import wkt
import pyproj as ppj

import grstbx

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(7.5, 41.5, width=100, height=100, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
bbox.to_crs(epsg=4326)
bbox.to_crs(epsg=32631)

opj = os.path.join
satdir = '/sat_data/satellite/sentinel2/L2A/GRS/31TGM'

image = 'S2*_v14.nc'
files = glob.glob(opj(satdir, image))
file=files[0]
product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                          decode_coords='all')

ppj.CRS.from_wkt(product.crs.wkt).to_epsg()
