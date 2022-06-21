
import os
import glob

from datetime import datetime as dt
import xarray as xr
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import grstbx


tile='31TEJ'
select =grstbx.select_files()
# if you need to change the root path
select.root='/home/harmel/Dropbox/satellite/S2'

select.list_tile(tile=tile,product='cnes')
files = select.list_file_path(('2022-01','2022-04-15'))


# central coordinate
lon, lat = 3.6, 43.4
# size in meter of the rectangle of interest
width, height = 2000, 2000

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
bbox = bbox.to_crs(epsg=32631)

# generate datacube
dc = grstbx.l2grs(files)
#dc.load(subset=bbox)
dc.Rrs.Rrs.isel(wl=2).plot.imshow(col='time',col_wrap=5,vmin=0,vmax=0.05,cmap=plt.cm.Spectral_r)
#bbox = bbox.to_crs(epsg=3857)
#dc.load(reproject=True, subset=bbox.bounds.values[0])
p_ = dc.datacube.isel(time=1)
masking_ = grstbx.masking(p_)
masking_.print_info()
masking_.get_mask(ndwi=True).compute()

nodata = masking_.get_mask(nodata=False).compute()
nodata.sum().compute()

from matplotlib_scalebar.scalebar import ScaleBar
bathyfile='/DATA/projet/magellium/malaigue/data/Bathymetrie_Region_LR_juillet-sept2009/isolignes_thau_20100625.shx'
bathy = gpd.read_file(bathyfile)
bathy =bathy.to_crs(3857)
bathy.CON