import os
import glob

from datetime import datetime as dt
import xarray as xr

opj = os.path.join
import geopandas as gpd
from shapely import wkt
import pyproj as ppj

import grstbx

ust = grstbx.utils.SpatioTemp()
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

products = []
for file in files:
    product = xr.open_dataset(file, chunks={'x': 512, 'y': 512},
                              decode_coords='all')
    product = product.drop(['lat', 'lon'])
    # add time dimension
    product = dc.add_time_dim(product)
    # ---------------------------------------------------
    # fix merging when MAJA flags are not always present
    # get names of variables
    varnames = pd.DataFrame(product.data_vars)
    vars_to_remove = varnames[0][varnames[0].str.contains("mask")]
    product = product.drop_vars(vars_to_remove)
    try:
        product = product.drop_vars('aot_maja')
    except:
        pass
    # ---------------------------------------------------
    products.append(product)

#product = xr.concat(products, dim='time').sortby('time')
masking_ = grstbx.Masking(product)
nodata = masking_.get_mask(nodata=False)
nodata.as_numpy()
##
p_ = product
bands=['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4',
                                      'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8',
                                      'Rrs_B8A']
wl = []
for name, band in p_[bands].items():
    wl.append(band.attrs['wavelength'])
Rrs = p_[bands]
Rrs = Rrs.to_array(dim='wl', name='Rrs').assign_coords(wl=wl).chunk({'wl': 1})