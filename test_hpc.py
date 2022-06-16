import os
import glob

from datetime import datetime as dt
import xarray as xr
import pandas as pd
import geopandas as gpd

import grstbx


tile='31TEJ'
select =grstbx.select_files()
select.list_tile(tile=tile)
files = select.list_file_path(('2022-01','2022-01-15'))


# central coordinate
lon, lat = 3.6, 43.4
# size in meter of the rectangle of interest
width, height = 18000, 18000

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
bbox = bbox.to_crs(epsg=32631)

# generate datacube
dc = grstbx.l2grs(files)
dc.load(subset=bbox.bounds.values[0])

#bbox = bbox.to_crs(epsg=3857)
#dc.load(reproject=True, subset=bbox.bounds.values[0])