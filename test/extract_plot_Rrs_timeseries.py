
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as xrio # activate the rio accessor

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

rc = {"font.family": "serif", "mathtext.fontset": "stix", 'font.size': 16, 'axes.labelsize': 18}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from obs2co_l2bgen import chl, spm,cdom,transparency

import grstbx

u = grstbx.utils
opj = os.path.join

grstbx.__version__

# extent in meters
extent = 200

idir='/DATA/projet/magellium/obs2co/data/coastal_waters'
stations=np.array(['4007015', '5010001', '6012001', '6012008', '5011201', '6013005',
       '6013023', '6013022', '7014001', '7014023', '7014024', '7015030',
       '7015031', '7015032', '60008192', '8017020', '9018007', '9021011',
       '60008355', '10023049', 'Sola', 'Frioul', 'PointB', 'Bouee13',
       'Comprian', 'Luc-sur-Mer', 'Antioche', 'Bizeux', 'PointC', 'Cézembre',
       'Sete', 'Smile', 'PointL', 'Astan', 'Estacade', 'Portzic', 'Eyrac',
       'pk86', 'pk52', 'pk30'])
insitu={}
for station in stations:
    insitu[station] = xr.open_dataset(opj(idir,'odatis_data_'+station+'.nc'))

def stats_from_roi(Rrs,group_coord ='wl',stat_coord='gridcell'):
    stacked = Rrs.dropna('time', thresh=0).stack(gridcell=["y", "x"]).dropna(
        'gridcell', thresh=0)
    stats = xr.Dataset({'median':stacked.groupby(group_coord).median(stat_coord)})
    stats['q25'] = stacked.groupby(group_coord).quantile(0.25,dim=stat_coord)
    stats['q75'] = stacked.groupby(group_coord).quantile(0.75,dim=stat_coord)
    stats['min'] = stacked.groupby(group_coord).min(stat_coord)
    stats['max'] = stacked.groupby(group_coord).max(stat_coord)
    stats['mean'] = stacked.groupby(group_coord).mean(stat_coord)
    stats['std'] = stacked.groupby(group_coord).std(stat_coord)
    stats['pix_num'] = stacked.count(stat_coord)
    return stats

file = opj(idir, 'fig','odatis_Rrs_extraction.pdf')
with (PdfPages(file) as pdf):
    for station in stations:
        insitu_ = insitu[station]
        file = glob.glob(
            '/home/harmel/Dropbox/Dropbox/satellite/S2/datacube/odatis/*2016-01-01_2024-01-01_' + station + '.nc')
        if len(file)>0:
            file=file[0]
        else:
            continue
        raster = xr.open_dataset(file, decode_coords='all').set_coords('spatial_ref')
        dc = grstbx.L2grs(file)
        dc.raster = raster

        lat, lon = float(insitu_.lat.mean()), float(insitu_.lon.mean())

        # clipping
        Rrs = grstbx.SpatioTemp.clip_raster(dc.raster.Rrs, lat, lon, extent)

        # masking
        Rrs_blue_avg = float(Rrs.sel(wl=443).mean())
        mask = xr.where((Rrs.sel(wl=490, method='nearest') < 0) | (Rrs.sel(wl=565, method='nearest') < 0), 1, 0)
        mask=mask.where((Rrs.sel(wl=443,method='nearest')<2*Rrs_blue_avg),2)
        Rrs=Rrs.where(mask==0)
        stats = stats_from_roi(Rrs.sel(wl=slice(400, 1000)))

        Npix = stats['pix_num'].isel(wl=2).squeeze()
        Npix_thresh = Npix.max()*0.8

        num_items = len(stats.time)
        col_wrap = 4
        rows = int(np.ceil(num_items / col_wrap))
        bands = [3, 2, 1]

        fig, axs = plt.subplots(nrows=rows, ncols=col_wrap,  sharey=True, figsize=(20, rows * 3.6))  # ,sharey=True
        fig.subplots_adjust(hspace=0.24, wspace=0.1,left=0.065, right=0.965,bottom=0.1, top=0.9)
        axs_ = axs.ravel()
        [axi.set_axis_off() for axi in axs_]
        for iax, (time_, group) in enumerate(stats.groupby('time')):
            color = 'black'
            if group['pix_num'].isel(wl=2).squeeze() < Npix_thresh:
                color = 'red'

            axs_[iax].set_axis_on()
            axs_[iax].minorticks_on()
            if iax >= (rows - 1) * col_wrap:
                axs_[iax].set_xlabel('$Wavelength\ (nm)$')
            date = group.time.dt.date.values
            axins = inset_axes(axs_[iax], width="45%", height="60%",
                               bbox_to_anchor=(.55, .4, 0.9, 0.9),
                               bbox_transform=axs_[iax].transAxes, loc=3)

            Rrs.sel(time=time_).isel(wl=bands).plot.imshow(robust=True, ax=axins)
            axins.set_title('')
            axins.set_axis_off()
            axs_[iax].axhline(y=0, color='k', lw=1)
            axs_[iax].plot(group.wl, group['median'], c='k')
            axs_[iax].plot(group.wl, group['mean'], c='red', ls='--')
            axs_[iax].fill_between(group.wl, group['q25'], group['q75'], alpha=0.3, color='grey')
            axs_[iax].set_title(date,color=color)
        for ax_ in axs[:,0]:
            ax_.set_ylabel(r'$R_{rs}\ (sr^{-1})$')

        plt.suptitle(station)
        pdf.savefig()
        plt.close()

file = opj(idir, 'fig', 'odatis_RGB.pdf')
with (PdfPages(file) as pdf):
    for station in stations:
        insitu_ = insitu[station]
        file = glob.glob(
            '/home/harmel/Dropbox/Dropbox/satellite/S2/datacube/odatis/*2016-01-01_2024-01-01_' + station + '.nc')
        if len(file) > 0:
            file = file[0]
        else:
            continue
        raster = xr.open_dataset(file, decode_coords='all').set_coords('spatial_ref')
        dc = grstbx.L2grs(file)
        dc.raster = raster
        bands = [3, 2, 1]
        coarsening = 1

        gamma = 1
        fig = (dc.raster.Rrs.isel(wl=bands)[:, :, ::coarsening, ::coarsening] ** (1 / gamma)).plot.imshow(col='time',
                                                                                                          col_wrap=4,
                                                                                                          robust=True)
        for ax in fig.axs.flat:
            ax.set(xticks=[], yticks=[])
            ax.set_ylabel('')
            ax.set_xlabel('')

        plt.suptitle(station)
        pdf.savefig()
        plt.close()


def cPOC_2(Rrs, p=[2.873, 0.945, 0.025]):
        ''' Ref: Tran, T.K.; Duforêt-Gaurier, L.; Vantrepotte, V.; Jorge, D.S.F.; Mériaux, X.; Cauvin, A.; Fanton d’Andon, O.; Loisel, H.
        Deriving Particulate Organic Carbon in Coastal Waters from Remote Sensing: Inter-Comparison Exercise
        and Development of a Maximum Band-Ratio Approach. Remote Sens. 2019, 11, 2849. https://doi.org/10.3390/rs11232849 '''

        ratio1 = Rrs.sel(wl=665, method='nearest') / Rrs.sel(wl=490, method='nearest')
        ratio2 = Rrs.sel(wl=665, method='nearest') / Rrs.sel(wl=565, method='nearest')
        ratio = np.log10(
            xr.concat([ratio1.assign_coords({'num': 1}), ratio1.assign_coords({'num': 2})], dim='num').max('num'))

        Xpoc = p[0] + p[1] * ratio + p[2] * ratio ** 2
        return 10 ** Xpoc

def plot_sat(param, ax, color='red',**kwargs):

    y = sat_median[param]
    yerr_low = y - sat_q25[param]
    yerr_up = sat_q75[param] - y
    ax.plot(x, y, marker='o', color=color,**kwargs)
    ax.errorbar(x, y, yerr=[yerr_low, yerr_up], marker='o', color=color)

stat_coord='gridcell'
file = opj(idir, 'fig', 'odatis_retrieval_timeseries.pdf')
with (PdfPages(file) as pdf):
    file = opj(idir, 'fig', 'odatis_matchup.pdf')
    with (PdfPages(file) as pdf_m):
        for station in stations:
            insitu_ = insitu[station]
            file = glob.glob(
                '/home/harmel/Dropbox/Dropbox/satellite/S2/datacube/odatis/*2016-01-01_2024-01-01_' + station + '.nc')
            if len(file) > 0:
                file = file[0]
            else:
                continue
            raster = xr.open_dataset(file, decode_coords='all').set_coords('spatial_ref')
            dc = grstbx.L2grs(file)
            dc.raster = raster

            lat, lon = float(insitu_.lat.mean()), float(insitu_.lon.mean())

            # clipping
            Rrs = grstbx.SpatioTemp.clip_raster(dc.raster.Rrs, lat, lon, extent)

            # masking
            Rrs_blue_avg = float(Rrs.sel(wl=443).mean())
            mask = xr.where((Rrs.sel(wl=490, method='nearest') < 0) | (Rrs.sel(wl=565, method='nearest') < 0), 1, 0)
            mask = mask.where((Rrs.sel(wl=443, method='nearest') < 2 * Rrs_blue_avg), 2)
            Rrs = Rrs.where(mask == 0)

            # QC and date rejection
            stats = stats_from_roi(Rrs.sel(wl=slice(400, 1000)))
            Npix = stats['pix_num'].isel(wl=2).squeeze()
            Npix_thresh = Npix.max() * 0.8
            QCdate = Npix.where(Npix > Npix.max() * 0.8, drop=True).time.values
            Rrs = Rrs.sel(time=QCdate)

            # L2Bgen
            l2a = xr.Dataset({'Rrs': Rrs})
            chl_prod = chl(l2a)
            chl_prod.process()

            # ----------------------
            # get SPM parameters
            # ----------------------
            spm_prod = spm(l2a)
            spm_prod.process()

            # ----------------------
            # get CDOM parameters
            # ----------------------
            cdom_prod = cdom(l2a)
            cdom_prod.process()

            # ----------------------
            # get transparency parameters
            # ----------------------
            trans_prod = transparency(l2a)
            trans_prod.process()


            # ----------------------
            # possibity to test other retrievals
            # ----------------------
            poc = cPOC_2(Rrs)
            poc.name = 'cPOC_2'

            # ----------------------
            # Merge parameters into xr.Dataset
            # ----------------------
            l2b = xr.merge([chl_prod.output, spm_prod.output, cdom_prod.output, trans_prod.output, poc])

            stacked = l2b.dropna('time', thresh=0).stack(gridcell=["y", "x"]).dropna('gridcell', thresh=0)
            stats = stats_from_roi(Rrs.sel(wl=slice(400, 1000)))

            sat_median = stacked.median(stat_coord)
            sat_std = stacked.std(stat_coord)

            sat_q25 = stacked.quantile(0.25, dim=stat_coord)
            sat_q75 = stacked.quantile(0.75, dim=stat_coord)


            x = sat_median.time.values
            fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15, 10), sharex=True)
            fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                                hspace=0.1, wspace=0.25)
            ii = 0
            param = 'Chla_OC2nasa'
            insitu_.chla.plot(ls='', marker='o', color='black', ax=axs[ii])
            plot_sat(param, axs[ii])
            axs[ii].minorticks_on()

            ii += 1
            insitu_.spm.plot(ls='', marker='o', color='black', ax=axs[ii])
            param =  'SPM_nechad'
            plot_sat(param, axs[ii],label='Nechad2010')
            param ='SPM_obs2co'
            plot_sat(param, axs[ii], color='violet',label='obs2co')
            axs[ii].legend()

            ii += 1
            param = 'cPOC_2'
            insitu_.poc.plot(ls='', marker='o', color='black', ax=axs[ii])
            plot_sat(param, axs[ii])
            plt.suptitle(station)


            pdf.savefig()
            plt.close()

            # ------------
            # matchup
            insitu_df = insitu_.rename({'date': 'time'}).sortby('time').to_dataframe()
            insitu_df['station'] = insitu_df['station'].astype(str).str.replace(' ', '')
            sat_df = sat_median.to_dataframe()
            sat_df = sat_df.merge(sat_std.to_dataframe(), on='time', suffixes=['_median', '_std'])

            matchup_4h = pd.merge_asof(
                sat_df, insitu_df, on="time", tolerance=pd.Timedelta("4h")
            )
            matchup_2d = pd.merge_asof(
                sat_df, insitu_df, on="time", tolerance=pd.Timedelta("2d")
            )

            params = {
                'chl': dict(insitu_var='chla', sat_var='Chla_OC2nasa_median', label='Chl\ a\ (\mu g \cdot L^{-1})'),
                'spm': dict(insitu_var='spm', sat_var='SPM_nechad_median', label='SPM\ (mg \cdot L^{-1})'),
                'poc': dict(insitu_var='poc', sat_var='cPOC_2_median', label='POC\ (\mu g \cdot L^{-1})')}
            u = grstbx.utils

            rows = 1
            cols = 3
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5 + 1.))  # ,sharey=True
            fig.subplots_adjust(hspace=0.24, wspace=0.25, left=0.065, right=0.965, bottom=0.12, top=0.9)
            axs_ = axs.ravel()
            [axi.set_axis_off() for axi in axs_]
            for ii, param in enumerate(['chl', 'spm', 'poc']):
                ax = axs[ii]
                insitu_var, sat_var, label = params[param].values()
                matchup_ = matchup_2d[[insitu_var, sat_var,sat_var.replace('median','std')]].dropna()
                if len(matchup_)==0:
                    continue
                ax.set_axis_on()
                x, y = matchup_[insitu_var].values.ravel(), matchup_[sat_var].values.ravel()

                all_values = [*x, *y]
                xmin, xmax = np.min([0, np.min(all_values)]) * 1.1, np.max(all_values) * 1.1

                im = matchup_4h[[insitu_var, sat_var]].dropna().plot.scatter(x=insitu_var, y=sat_var, ax=ax, zorder=3)
                im2 = matchup_.plot.scatter(x=insitu_var, y=sat_var, color='red', alpha=0.5, ax=ax, zorder=2)
                ax.errorbar(x, y,
                            #            xerr=matchup[insitu_var.replace('mean','std')].values.ravel(),
                            yerr=matchup_[sat_var.replace('median', 'std')].values.ravel(),
                            color='gray', fmt='none', lw=1, capsize=3)  # ecolor=resdf.spm_norm, zorder=9)

                u.plot.add_stats(x, y, ax, label=True)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([xmin, xmax])
                ax.minorticks_on()
                u.plot.set_layout(ax)

                ax.set_title('Matchup GRS / ODATIS')
                ax.set_xlabel(r'$In\ situ\ ' + label + '$')
                ax.set_ylabel(r'$Satellite\ ' + label + '$')
            plt.suptitle(station)

            pdf_m.savefig()
            plt.close()





matchup_4h,matchup_2d=[],[]
for station in stations:
    insitu_ = insitu[station]
    file = glob.glob(
        '/home/harmel/Dropbox/Dropbox/satellite/S2/datacube/odatis/*2016-01-01_2024-01-01_' + station + '.nc')
    if len(file) > 0:
        file = file[0]
    else:
        continue
    raster = xr.open_dataset(file, decode_coords='all').set_coords('spatial_ref')
    dc = grstbx.L2grs(file)
    dc.raster = raster
    # TODO to be removed for GRS v2.1
    dc.raster = dc.raster.rename({'o2_band': 'wv_band'})

    Rrs= dc.raster.Rrs

    lat, lon = float(insitu_.lat.mean()), float(insitu_.lon.mean())


    # masking
    Rrs_blue_avg = float(Rrs.sel(wl=443).mean())
    mask = xr.where((Rrs.sel(wl=490, method='nearest') < 0) | (Rrs.sel(wl=565, method='nearest') < 0), 1, 0)
    mask = mask.where((Rrs.sel(wl=443, method='nearest') < 2 * Rrs_blue_avg), 2)
    mask = mask.where((Rrs.sel(wl=443, method='nearest') < 3 * Rrs_blue_avg), 3)
    mask = mask.where((dc.raster.wv_band) < 0.1, 4)
    Rrs = Rrs.where(mask == 0)

    # clipping
    Rrs = grstbx.SpatioTemp.clip_raster(Rrs, lat, lon, extent)

    # QC and date rejection
    stats = stats_from_roi(Rrs.sel(wl=slice(400, 1000)))
    Npix = stats['pix_num'].isel(wl=2).squeeze()
    Npix_thresh = Npix.max() * 0.8
    QCdate = Npix.where(Npix > Npix.max() * 0.8, drop=True).time.values
    Rrs = Rrs.sel(time=QCdate)

    # L2Bgen
    l2a = xr.Dataset({'Rrs': Rrs})
    chl_prod = chl(l2a)
    chl_prod.process()

    # ----------------------
    # get SPM parameters
    # ----------------------
    spm_prod = spm(l2a)
    spm_prod.process()

    # ----------------------
    # get CDOM parameters
    # ----------------------
    cdom_prod = cdom(l2a)
    cdom_prod.process()

    # ----------------------
    # get transparency parameters
    # ----------------------
    trans_prod = transparency(l2a)
    trans_prod.process()

    # ----------------------
    # possibity to test other retrievals
    # ----------------------
    poc = cPOC_2(Rrs)
    poc.name = 'cPOC_2'

    # ----------------------
    # Merge parameters into xr.Dataset
    # ----------------------
    l2b = xr.merge([chl_prod.output, spm_prod.output, cdom_prod.output, trans_prod.output, poc])

    stacked = l2b.dropna('time', thresh=0).stack(gridcell=["y", "x"]).dropna('gridcell', thresh=0)
    stats = stats_from_roi(Rrs.sel(wl=slice(400, 1000)))

    sat_median = stacked.median(stat_coord)
    sat_std = stacked.std(stat_coord)

    sat_q25 = stacked.quantile(0.25, dim=stat_coord)
    sat_q75 = stacked.quantile(0.75, dim=stat_coord)


    # ------------
    # matchup
    insitu_df = insitu_.rename({'date': 'time'}).sortby('time').to_dataframe()
    insitu_df['station'] = insitu_df['station'].astype(str).str.replace(' ', '')
    sat_df = sat_median.to_dataframe()
    sat_df = sat_df.merge(sat_std.to_dataframe(), on='time', suffixes=['_median', '_std'])

    matchup_4h.append( pd.merge_asof(
        sat_df, insitu_df, on="time", tolerance=pd.Timedelta("4h")
    ))
    matchup_2d.append( pd.merge_asof(
        sat_df, insitu_df, on="time", tolerance=pd.Timedelta("2d")
    ))

matchup_2d = pd.concat(matchup_2d)
matchup_2d = matchup_2d.set_index('time')

# !!!!!!!!!!!!!!!!!!!!!!
# Warning remove outliers for spm> 1000
matchup_2d=matchup_2d.drop(matchup_2d[matchup_2d.spm>1000].index)
matchup_2d=matchup_2d.drop(matchup_2d[matchup_2d.poc>4000].index)

params = {'chl':dict(insitu_var='chla',sat_var='Chla_OC2nasa_median',label='Chl\ a\ (\mu g \cdot L^{-1})'),
          'spm':dict(insitu_var='spm',sat_var='SPM_nechad_median',label='SPM\ (mg \cdot L^{-1})'),
          'poc':dict(insitu_var='poc',sat_var='cPOC_2_median',label='POC\ (\mu g \cdot L^{-1})')}
u = grstbx.utils

rows=2
cols=3
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*5+1, rows * 5+1.))  # ,sharey=True
fig.subplots_adjust(hspace=0.275, wspace=0.275,left=0.065, right=0.965,bottom=0.08, top=0.95)
for ii, param in enumerate(['chl','spm','poc']):
    for irow in range(rows):
        ax=axs[irow,ii]
        insitu_var,sat_var,label=params[param].values()
        matchup_=matchup_2d[[insitu_var,sat_var,sat_var.replace('median','std'),'station']].dropna()
        x,y=matchup_[insitu_var].values.ravel(),matchup_[sat_var].values.ravel()

        all_values = [*x,*y]
        xmin,xmax=np.min([0,np.min(all_values)])*1.1,np.max(all_values)*1.1

        ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, 40)))
        for station,m_ in matchup_.groupby('station'):
            ax.plot(m_[insitu_var],m_[sat_var],ls='',marker='o',ms=3.5,alpha=0.75,zorder=2)
        #im2 = matchup_.groupby('station').plot.scatter(x=insitu_var,y=sat_var,ax=ax)
        ax.errorbar(x,y,
        #            xerr=matchup[insitu_var.replace('mean','std')].values.ravel(),
                    yerr=matchup_[sat_var.replace('median','std')].values.ravel(),
                    color='gray',fmt='none',lw=1, capsize=3,zorder=1)# ecolor=resdf.spm_norm, zorder=9)

        if irow ==0:
            u.plot.add_stats(x,y,ax,label=True)

        ax.set_xlim([xmin,xmax])
        ax.set_ylim([xmin,xmax])
        ax.minorticks_on()
        u.plot.set_layout(ax)

        ax.set_title('Matchup GRS / ODATIS')
        ax.set_xlabel(r'$In\ situ\ '+label+'$')
        ax.set_ylabel(r'$Satellite\ '+label+'$')

limits=[[1e-2,200],[1e-2,2000],[1,5000]]
for icol in range(cols):
    axs[1,icol].loglog()
    axs[1,icol].set_xlim(*limits[icol] )
    axs[1, icol].set_ylim(*limits[icol] )
    u.plot.set_layout(axs[1, icol])

plt.savefig(opj(idir, 'fig', 'obs2co_odatis_matchups_without_outliers.png'),dpi=300)#
xmin,xmax=-0.5,30
axs[0,1].set_xlim([xmin, xmax])
axs[0,1].set_ylim([xmin, xmax])
plt.savefig(opj(idir, 'fig', 'obs2co_odatis_matchups_without_outliers_zoom.png'),dpi=300)#
