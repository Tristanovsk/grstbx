
import glob
import os

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import pandas as pd
import geopandas as gpd

import xarray as xr
import rioxarray  # activate the rio accessor
import datetime as dt
import grstbx
from grstbx import visual

plt.rcParams.update({'font.size': 16})

rc = {"font.family": "serif", "mathtext.fontset": "stix", 'font.size': 16, 'axes.labelsize': 18}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

u = grstbx.utils
opj = os.path.join
#xr.backends.list_engines()
grstbx.__version__

idir='/DATA/projet/magellium/obs2co/data/coastal_waters'
ifile=opj(idir,'mc_msi_polymer_acolite.xlsx')

df=pd.read_excel(ifile) #,index_col=[3,4,5,6])
df = df.drop_duplicates()
df['date']=df['date']+pd.to_timedelta(df['insitu_time'],'hours')
df=df.drop('insitu_time',axis=1)
df=df[df.depth <= 3.]
df = df.set_index(['date','lon','lat','station'])

# get duplicated samples (same date time) and average them
duplicated = df[df.index.duplicated(keep=False)]
str_duplicated=df[df.index.duplicated(keep='first')][[ 'station_id', 'source', 'depth_id','q_chla','q_poc', 'q_spm', 'q_turb' ]]
duplicated= duplicated[[ 'depth', 'chla',
        'poc', 'spm', 'turb']].groupby(['date','lon','lat']).mean()
dupes = str_duplicated.merge(duplicated,left_index=True, right_index=True)
# remove duplicated
df=df[~df.index.duplicated(keep=False)]

# merge averaged duplicated into main dateframe
df=pd.concat([df,dupes])

# export data into netcdf for each station
for station,df_ in df.groupby('station'):
    xarr=df_.reset_index().set_index(['date']).to_xarray()
    station=str(station).replace(' ','')
    ofile=opj(idir,'odatis_data_'+station+'.nc')
    if os.path.exists(ofile):
        os.remove(ofile)
    xarr.to_netcdf(ofile)

#---------------------------
# Plot maps and histograms
#---------------------------

import cartopy.crs as ccrs
import cartopy.feature as cfeature

cm = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['mediumblue','steelblue','ivory','green','darkgreen'
                                                    ])
#cm = plt.cm.get_cmap('RdYlBu_r')
#cm = plt.cm.get_cmap('ocean_r')
cms=[cmocean.cm.delta,mpl.cm.Spectral_r,cmocean.cm.diff,cmocean.cm.balance]
xlabels=[r'$Chl-a\ (\mu g\cdot L^{-1})$',r'$SPM\ (mg\cdot L^{-1})$',r'$POC\ (\mu g\cdot L^{-1})$']
vmaxs=[8,1000,6000]
fig = plt.figure(figsize=(17,12))
for ii,param in enumerate(['chla','spm','poc']):
    ax = fig.add_subplot(2, 3, ii+1, projection=ccrs.PlateCarree())
    ax.set_extent([-5, 8, 41.25, 52])

    ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.RIVERS.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    df_ = df.reset_index()

    colors=df_[param]#,vmin=0.1, vmax=vmaxs[ii]
    sc = ax.scatter(df_.lon, df_.lat,  c=colors, norm=mpl.colors.LogNorm(),s=35, alpha=1,cmap=cms[ii],zorder=6)
    plt.colorbar(sc,location='top',label=xlabels[ii])
    #ax.set_title(param)
    ax = fig.add_subplot(2, 3, ii + 4)
    colors.hist(ax=ax,bins=20,rwidth=0.9,
                   color='#607c8e')
    ax.text(0.65, 0.95,
            'N = {:.0f}'.format(colors.count()) +
            '\nmean = {:.1f} '.format(colors.mean()) +
            '\nstd = {:.1f} '.format(colors.std()) +
            '\nmin = {:.1f} '.format(colors.min())+
            '\nmax = {:.1f} '.format(colors.max()),

            bbox=dict(boxstyle="round", fc='1'), fontsize=11, horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.semilogy()
    ax.set_xlabel(xlabels[ii])
plt.suptitle('ODATIS data for coastal validation')
plt.savefig('test/odatis_dataset_maps_histo.png',dpi=300)


#---------------------------
# Plot time series
#---------------------------

fig,axs = plt.subplots(3,1,figsize=(16,13),sharex=True)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.265,top=0.97,hspace=.1, wspace=0.05)
tsdf = df.reset_index().set_index('date').groupby('station')
for ii,param in enumerate(['chla','spm','poc']):
    axs[ii].set_prop_cycle('color', plt.cm.Spectral_r(np.linspace(0, 1, 42)))
    tsdf[param].plot(ls='',marker='o',ms=3.5,alpha=0.75,ax=axs[ii])
    axs[ii].semilogy()
    axs[ii].set_ylabel(xlabels[ii])
    axs[ii].minorticks_on()
axs[ii].set_xlabel('')
axs[ii].legend(title='$Sites$',loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  fancybox=True, shadow=True, ncol=7, handletextpad=0.1, fontsize=12)
plt.savefig('test/odatis_dataset_timeseries.png',dpi=300)


xarr.chla.plot(marker='o',ls='',col='station',col_wrap=3)
xarr.poc.plot(marker='o',ls='',col='station',col_wrap=3)
xarr.spm.plot(marker='o',ls='',col='station',col_wrap=3)
xarr.turb.plot(marker='o',ls='',col='station',col_wrap=3)


#---------------------------
# Format GRS list
#---------------------------

####
# find tiles
# try with kml failed
# s2_tiles_file='/DATA/Satellite/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# gpd.read_file(s2_tiles_file, driver='KML')

xarr=df.reset_index(['lon','lat']).to_xarray()
station = xarr.mean('date')

df_station=station[['lon','lat']].to_dataframe()

import simplekml
kml = simplekml.Kml()
df_station.reset_index().apply(lambda X: kml.newpoint(name=X["station"], coords=[( X["lon"],X["lat"])]) ,axis=1)
#df_station.reset_index().to_csv("test/odatis_stations.csv")
kml.save(path = "test/odatis_stations.kml")
s2tiles=pd.read_csv("test/odatis_stations_s2tiles.csv")

df = df.reset_index()
df_complete = df.merge(s2tiles[['station','s2_tile']],on='station')


# construct list file for GRS processing
date_tile = df_complete[['date','s2_tile']]
date_tile['date'] = date_tile['date'].dt.date
date_tile=date_tile.drop_duplicates(keep='first')

# process (yes if 1),Site Name,start_date,end_date,satellite,tile,resolution (m),flag
# 0,_v20,2021-12-01,2021-12-31,S2,31TFJ,20,1
# ...
grs_list = pd.DataFrame(columns=['process (yes if 1)','Site Name','start_date','end_date','satellite','tile','resolution (m)','flag'])
grs_list['start_date']=date_tile['date']
grs_list['end_date']=date_tile['date']
grs_list['tile']=date_tile['s2_tile']
grs_list['process (yes if 1)']=1
grs_list['Site Name']='_v20'
grs_list['satellite']='S2'
grs_list['resolution (m)']=20
grs_list['flag']=1

grs_list.to_csv('test/list_grs_odatis.csv',index=False)

