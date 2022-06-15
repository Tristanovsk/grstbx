import os
import glob
import geopandas as gpd
import grstbx
from datetime import datetime as dt
import xarray as xr

opj = os.path.join
satdir = '/sat_data/satellite/sentinel2/L2A/GRS/31TGM'

image = 'S2*_v14.nc'
files = glob.glob(opj(satdir, image))

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(6.6, 46.45, width=1000, height=1000, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)

bbox = bbox.to_crs(epsg=32631)
print(bbox)
bbox.to_crs(epsg=3857)

dc = grstbx.l2grs(files,)
dc.load(subset=bbox.bounds.values[0])

dc.Rrs.Rrs.plot(col='wl',row='time',vmin=0)

def add_time_dim(xda):
    time=[dt.strptime(xda.attrs['start_date'], '%d-%b-%Y %H:%M:%S.%f')]
    xda = xda.assign_coords(time=time)
    #xda = xda.expand_dims('time')
    return xda

product = xr.open_mfdataset(files[0:1], chunks={'x': 512, 'y': 512},
                            decode_coords='all',combine='nested',
                            concat_dim="time",preprocess = add_time_dim,
                            parallel=True)