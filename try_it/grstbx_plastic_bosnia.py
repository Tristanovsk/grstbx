
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd

import xarray as xr
import rioxarray  # activate the rio accessor

#import cartopy
import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.utf8')
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')

import pyproj as ppj
import rasterio
from affine import Affine
from shapely.geometry import box,Point, mapping
from shapely.ops import transform

import dask.distributed
cluster = dask.distributed.LocalCluster()
client = dask.distributed.Client(cluster)
client
import grstbx
from grstbx import visual


u = grstbx.utils
opj = os.path.join
#xr.backends.list_engines()
grstbx.__version__



# ## Set images to play with

tile='34TCP'
# central coordinate
lon, lat = 19.28,43.758
# size in meter of the rectangle of interest
width, height = 2500, 1600
select =grstbx.select_files()
# if you need to change the root path
#select.root='/media/harmel/TOSHIBA EXT/data/satellite'
select.list_tile(tile=tile,product='S2-L2GRS',pattern='*v14.nc')
files = select.list_file_path(('2021-01-01','2021-12-31'))
print(select.list.iloc[:,:-1])


# In[11]:


import pyproj
pyproj.datadir.get_data_dir()
pyproj.datadir.set_data_dir('/work/scratch/harmelt/envs/grstbx/share/proj')


# ## Load and subset image series
ust = grstbx.utils.spatiotemp()
box = ust.wktbox(lon,lat, width=width, height=height, ellps='WGS84')
bbox = gpd.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
# reproject lon lat in xy coordinates
#bbox = bbox.to_crs(epsg=32720)
bbox.to_crs(4326).bounds

# generate datacube
dc = grstbx.l2grs(files)
dc.load(subset=bbox)



# ## Check data/metadata
dc.Rrs


# ## Check flags and masking

pd.DataFrame.from_dict(dc.datacube.metadata.attrs,orient='index')


# ## Check flags and masking


masking_ = grstbx.masking(dc.datacube)
masking_.print_info()


# In[13]:


mask = (masking_.get_mask(high_nir=True) | masking_.get_mask(hicld=True)) 
aspect=1.5
mask.plot.imshow(col='time',col_wrap=5,vmin=0,cmap=plt.cm.binary_r,aspect=aspect)


# ## Mask datacube
# Mask pixels from chosen flags and remove empty dates

# In[14]:


Rrs_masked = dc.Rrs.Rrs.where(mask==0).dropna('time','all')
BRDFg = dc.datacube.BRDFg.where(mask==0).dropna('time','all')
BRDFg = BRDFg.where(masking_.get_mask(ndwi=False),drop=True)


# In[15]:


Rrs_masked


# In[16]:


Rrs_masked.isel(wl=2).plot(col='time',col_wrap=5,vmin=0,vmax=0.03, robust=True,cmap=plt.cm.Spectral_r,aspect=1.6)


# ## **Check spectral datacube** (i.e., Remote Sensing Reflectance, R<sub>rs</sub>, sr<sup>-1</sup>)

# To quickly check your data visually, you can use the *visual* module of *grstbx*

# In[17]:


visual.image_viewer().Rrs_date(Rrs_masked)


# ## **Fast checking of the RGB images**

# In[18]:


bands=[4,2,1]
bands=[3,2,1]
Rrs_masked.isel(wl=bands).plot.imshow(col='time', col_wrap=6,robust=True,aspect=aspect)
plt.savefig('bosnia_test.png')

# In[ ]:





# In[19]:


import datashader as ds
from datashader import transfer_functions as tf 
from colorcet import palette


shaded = []
for name, raster in Rrs_masked.isel(wl=2).groupby('time'):
    img = tf.shade(raster.squeeze(),cmap=cc.kbc)
    img.name = str(name)
    shaded.append(img)

imgs = tf.Images(*shaded)
imgs.num_cols = 5
imgs


# In[20]:



shaded = []
for name, raster in Rrs_masked.isel(wl=5).groupby('time'):
    img = tf.shade(raster.squeeze(),cmap=cc.kbc)
    img.name = str(name)
    shaded.append(img)

imgs = tf.Images(*shaded)
imgs.num_cols = 5
imgs


# In[ ]:





# In[21]:


from holoviews import opts

opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='binary_r', width=800, height=700),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=900),
    opts.Overlay(show_legend=True))
# set the parameter for spectra extraction
hv.extension('bokeh')
pn.extension()

raster = Rrs_masked#.isel(time=-1,drop=True)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= 'RdBu_r',colorbar=True)#.hist(bin_range=(0,0.02) ) 
widget = pn.widgets.RangeSlider(start=0, end=0.1,step=0.001)

jscode = """
    color_mapper.low = cb_obj.value[0];
    color_mapper.high = cb_obj.value[1];
"""
link = widget.jslink(im, code={'value': jscode})
pn.Column(widget, im)


# In[22]:



param = 'Rrs'
third_dim = 'wl'
raster = Rrs_masked.isel(time=-1,drop=True)
wl= raster.wl.data
Nwl = len(wl)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= 'RdBu_r',colorbar=True,clim=(0,None)).hist(bin_range=(0,0.02) ) 

polys = hv.Polygons([])
box_stream = hv.streams.BoxEdit(source=polys)
dmap, dmap_std=[],[]

def roi_curves(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', 'Rrs')})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]= hv.Curve((wl,mean[param]),'Wavelength (nm)', 'Rrs') 

    return hv.NdOverlay(curves)


# a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
def roi_spreads(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', 'Rrs')})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]=  hv.Spread((wl,mean[param],std[param]),fill_alpha=0.3)

    return hv.NdOverlay(curves)

mean=hv.DynamicMap(roi_curves,streams=[box_stream])
std =hv.DynamicMap(roi_spreads, streams=[box_stream])    
hlines = hv.HoloMap({wl[i]: hv.VLine(wl[i]) for i in range(Nwl)},third_dim )


# In[23]:



# visualize and play
graphs = ((mean* std *hlines).relabel('Rrs'))
layout = (im * polys +graphs    ).opts(
    opts.Curve(width=600, framewise=True,xlim=(400,1000)), 
    opts.Polygons(fill_alpha=0.2, color='green',line_color='black'), 
    opts.VLine(color='black')).cols(2)
layout 


# # Check surface rugosity via sunglint BRDF

# In[24]:



raster = BRDFg#.isel(time=-1,drop=True)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= 'gray',colorbar=True)#.hist(bin_range=(0,0.02) ) 
widget = pn.widgets.RangeSlider(start=0, end=0.01,step=0.001)

jscode = """
    color_mapper.low = cb_obj.value[0];
    color_mapper.high = cb_obj.value[1];
"""
link = widget.jslink(im, code={'value': jscode})
pn.Column(widget, im)


# ## Check blue over green ratio for Chl retrieval with OC2 from NASA
# $log_{10}(chlor\_a) = a_0 + \sum\limits_{i=1}^4 a_i \left(log_{10}\left(\frac{R_{rs}(\lambda_{blue})}{R_{rs}(\lambda_{green})}\right)\right)^i$

# In[25]:


# NASA OC2 fro MODIS; bands 488, 547 nm
a = [0.2500,-2.4752,1.4061,-2.8233,0.5405]
# NASA OC2 for OCTS; bands 490, 565 nm
a = [0.2236,-1.8296,1.9094,-2.9481,-0.1718]

ratio = np.log10(Rrs_masked.isel(wl=1)/Rrs_masked.isel(wl=2))
logchl=0
for i in range(len(a)):
    logchl+=a[i]*ratio**i
chl = 10**(logchl)
chl.name='chl in mg.m-3 from OC2'


# Set range of valid values

# In[26]:


chl = chl.where((chl >= 0) & (chl <= 80))
chl.persist()


# In[27]:


visual.image_viewer().param_date(chl,cmap='bgyw')


# In[ ]:





# In[ ]:


raster = chl

shaded = []
for name, raster in chl.groupby('time'):
    img = tf.shade(raster.squeeze(),cmap=cc.bgyw, span=(0,10),how='log')
    img.name = str(name)
    shaded.append(img)

imgs = tf.Images(*shaded)
imgs.num_cols = 4
imgs


# # CDOM retrieval based on Brezonik et al, 2015
# 

# In[ ]:


a = [1.872,-0.83]
acdom = np.exp(a[0] + a[1] * np.log(Rrs_masked.isel(wl=1)/Rrs_masked.isel(wl=5)))
acdom.name='CDOM absorption at 440 nm-1'
acdom= acdom.where((acdom >= 0) & (acdom <= 10))
acdom.persist()


# In[ ]:


visual.image_viewer().param_date(acdom,cmap='bgyw')


# # Total suspended particulate matter (SPM) from Nechad et al., 2010, 2016 formulation
# spm in mg/L

# In[ ]:


a = [610.94*np.pi, 0.2324/np.pi]
Rrs_ = Rrs_masked.isel(wl=3)
spm = a[0] * Rrs_ / (1 - ( Rrs_/ a[1]))
spm.name='CDOM absorption at 440 nm-1'
spm= spm.where((spm >= 0) & (spm <= 150))
spm.persist()


# In[ ]:


visual.image_viewer().param_date(spm,cmap='bgyw')


# In[ ]:



shaded = []
for name, raster in dc.Rrs.Rrs.isel(time=-1).groupby('wl'):
    img = tf.shade(raster,cmap=cc.kbc)
    img.name = '{:.2f}'.format(name)+' nm'
    shaded.append(img)

imgs = tf.Images(*shaded)
imgs.num_cols = 3
imgs


# In[ ]:


shaded = []
for name, raster in BRDFg.groupby('time'):
    img = tf.shade(raster,cmap=cc.gray, span=(0,0.025),how='log')
    img.name = str(name)
    shaded.append(img)

imgs = tf.Images(*shaded)
imgs.num_cols = 4
imgs


# # Play with time series

# In[ ]:


raster = spm

param = raster.name
third_dim = 'time'
time= raster.time.data
Ntime = len(time)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= 'RdBu_r',colorbar=True,clim=(0,10))#.hist(bin_range=(0,0.02) ) 

polys = hv.Polygons([])
box_stream = hv.streams.BoxEdit(source=polys)
dmap, dmap_std=[],[]

def roi_curves(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'time', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.nanmean).data
        print(mean)
        #std = selection.aggregate(third_dim, np.std).data
        time = mean[third_dim]

        curves[i]= hv.Curve((time,mean[param]),'time', param) 

    return hv.NdOverlay(curves)


# a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
def roi_spreads(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'time', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.nanmean).data
        std = selection.aggregate(third_dim, np.nanstd).data
        time = mean[third_dim]

        curves[i]=  hv.Spread((time,mean[param],std[param]),fill_alpha=0.3)

    return hv.NdOverlay(curves)

mean=hv.DynamicMap(roi_curves,streams=[box_stream])
std =hv.DynamicMap(roi_spreads, streams=[box_stream])    


# In[ ]:


# visualize and play
graphs = ((mean*std ).relabel(param))
layout = (im * polys +graphs    ).opts(
    opts.Curve(width=600, framewise=True), 
    opts.Polygons(fill_alpha=0.2, color='green',line_color='black'), 
    ).cols(2)
layout 


# In[ ]:


selection = ds.select(x=(547000,547200), y=(4802000, 4803000))
mean = selection.aggregate(third_dim, np.nanmean).data
std = selection.aggregate(third_dim, np.std).data
time = mean[third_dim]
curves={}
for i in [0,1,2]:
    curves[i]=hv.Curve((time,mean[param]),'time', param)

hv.DynamicMap(hv.NdOverlay(curves)).opts(width=600)


# In[ ]:





# In[ ]:




