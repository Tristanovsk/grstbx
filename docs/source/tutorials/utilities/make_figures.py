"""
Figures of the Utilities tutorial page (docs/source/tutorials/utilities.md).

The input data are not distributed with grstbx: set the paths below to regenerate the figures with your own
products, then run ``python make_figures.py`` from this folder.
"""
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import grstbx
from grstbx import Data, Dem, Irradiance, Plotting, SpatioTemp, plot_rgb, utm_projection
from grstbx.app import find_products
from grstbx.driver import open_zarr_image

warnings.filterwarnings('ignore')

# Sentinel-2 L2A product of the Loire estuary (Zarr pyramid)
LOIRE = ('/data/satellite/Sentinel-2/test/S2A_MSIL2AGRS_20170412T110621_N0500_R137_T30TWT_20231111T110836_V3.0.1dev/'
         'S2A_MSIL2AGRS_20170412T110621_N0500_R137_T30TWT_20231111T110836_V3.0.1dev.zarr')
# L2A products of tile 31TEJ (Cevennes, Gulf of Lion), tree <tile>/YYYY/MM/DD/<product>
TERRAIN_DIR = '/data/satellite/Sentinel-2/L2A_zarr/31TEJ'
# HYPERNETS spectra of the BEFR station and matchups of the L2A validation tutorial
HYPERNETS = '/DATA/hypernets/BEFR/BEFR_hypernets.nc'
MATCHUPS = '/data/satellite/Sentinel-2/subset/hypernets/datacube/BEFR_2021-01-01_2023-04-30_L2A_matchup_stats.nc'

ODIR = os.path.dirname(os.path.abspath(__file__))
DPI = 100


def save(fig, name):
    fig.savefig(os.path.join(ODIR, name), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('written', name)


def geometry():
    """wktbox and clip_raster on the Loire estuary."""
    lon, lat = -2.30, 47.24                       # Loire river plume, off Saint-Nazaire
    preview = open_zarr_image(LOIRE, level=2)    # 4 x coarser level, for the overview
    box = SpatioTemp().wktbox(lon, lat, width=5000, height=5000)
    aoi = grstbx.aoi_from_box(lon, lat, width=5000, height=5000)

    product = open_zarr_image(LOIRE, level=0)
    clipped = SpatioTemp.clip_raster(product[['Rrs']], lat, lon, extent_m=5000).load()

    proj = utm_projection(product.rio.crs)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={'projection': proj})
    plot_rgb(preview, ax=axs[0], aoi=aoi)
    axs[0].set_title('Sentinel-2 L2A, 2017-04-12: 5 km box of wktbox (red)')
    plot_rgb(clipped, ax=axs[1])
    axs[1].set_title(f'clip_raster: {clipped.sizes["x"]} x {clipped.sizes["y"]} pixels at '
                     f'{abs(clipped.rio.resolution()[0]):.0f} m')
    for ax in axs:
        ax.gridlines(draw_labels=['bottom', 'left'], lw=0.3, x_inline=False, y_inline=False)
    print('  wktbox:', box)
    save(fig, 'geometry.png')


def irradiance():
    """Extraterrestrial solar irradiance and its values at the Sentinel-2 bands."""
    irr = Irradiance()
    irr.load_F0()
    wl_s2 = np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190])
    F0 = irr.get_F0(wl_s2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(irr.F0df.wl, irr.F0df.F0, lw=0.6, c='k', label='Thuillier et al. (2003), 0.3 nm')
    ax.plot(wl_s2, F0, 'o', c='tab:red', ms=6, label='get_F0 at the Sentinel-2 bands')
    ax.set(xlim=(300, 2500), ylim=(0, None), xlabel='Wavelength (nm)',
           ylabel='F$_0$ (mW m$^{-2}$ nm$^{-1}$)')
    ax.legend()
    ax.grid(lw=0.3)
    save(fig, 'irradiance.png')


def remove_wavelengths():
    """remove_wl_dataarray on a HYPERNETS spectrum."""
    hypernets = xr.open_dataset(HYPERNETS)
    # day with the most spectra in the first half of October 2021
    period = hypernets['reflectance'].sel(time=slice('2021-10-01', '2021-10-15'))
    day = period.time.dt.floor('1D').to_series().value_counts().idxmax()
    spectrum = (period.sel(time=day.strftime('%Y-%m-%d')) / np.pi).median('time')
    removed = [(683, 695), (755, 770), (810, 835)]   # O2-B, O2-A and water vapour bands
    cleaned = Data.remove_wl_dataarray(spectrum, wl_to_remove=removed)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(spectrum.wl, spectrum, c='lightgrey', lw=3, label=f'HYPERNETS ({spectrum.sizes["wl"]} wavelengths)')
    ax.plot(cleaned.wl, cleaned, '.', c='tab:blue', ms=2,
            label=f'remove_wl_dataarray ({cleaned.sizes["wl"]} wavelengths kept)')
    for wl_min, wl_max in removed:
        ax.axvspan(wl_min, wl_max, color='tab:red', alpha=0.15, lw=0)
    ax.set(xlabel='Wavelength (nm)', ylabel='$R_{rs}$ (sr$^{-1}$)',
           title=f'Berre lagoon (BEFR), median of {day:%Y-%m-%d}; removed ranges in red')
    ax.legend()
    ax.grid(lw=0.3)
    save(fig, 'remove_wl.png')


def terrain():
    """Dem.compute_dem_attributes on the DEM of an L2A product."""
    path = find_products(TERRAIN_DIR).path.iloc[0]
    product = open_zarr_image(path, level=2)     # 80 m: enough to show the relief of a whole tile
    dem = product.dem.load()
    sza, azi = product.attrs['mean_solar_zenith_angle'], product.attrs['mean_solar_azimuth']
    relief = Dem.compute_dem_attributes(dem, sza, azi)
    relief3 = Dem.compute_dem_attributes(dem, sza, azi, z_factor=3)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4.6))
    dem.plot.imshow(ax=axs[0], cmap='terrain', vmin=0, cbar_kwargs={'label': 'altitude (m)'})
    np.degrees(relief.slope).plot.imshow(ax=axs[1], cmap='viridis', vmax=40,
                                          cbar_kwargs={'label': 'slope (deg)'})
    relief.shaded.plot.imshow(ax=axs[2], cmap='Greys_r', vmin=0.6, vmax=1,
                              cbar_kwargs={'label': r'$\cos\theta_i$'})
    relief3.shaded.plot.imshow(ax=axs[3], cmap='Greys_r', vmin=0, vmax=1,
                               cbar_kwargs={'label': r'$\cos\theta_i$'})
    titles = ['DEM of the L2A product', 'slope',
              f'illumination, SZA {sza:.0f}°, azimuth {azi:.0f}°', 'illumination, z_factor=3']
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_aspect('equal')
    date = find_products(TERRAIN_DIR).time.iloc[0]
    fig.suptitle(f'Tile 31TEJ (Cevennes, Gulf of Lion), Sentinel-2 L2A of {date:%Y-%m-%d}, 80 m', y=1.02)
    save(fig, 'terrain.png')


def maps():
    """utm_projection and plot_rgb: true and false colour."""
    preview = open_zarr_image(LOIRE, level=2)
    aoi = grstbx.aoi_from_box(-2.25, 47.22, width=15000, height=12000)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5),
                            subplot_kw={'projection': utm_projection(preview.rio.crs)})
    plot_rgb(preview, ax=axs[0], aoi=aoi)
    axs[0].set_title('plot_rgb: true colour (665, 560, 490 nm), gamma 0.5')
    plot_rgb(preview, ax=axs[1], bands=(865, 665, 560), gamma=1)
    axs[1].set_title('false colour (865, 665, 560 nm), gamma 1')
    for ax in axs:
        ax.gridlines(draw_labels=['bottom', 'left'], lw=0.3, x_inline=False, y_inline=False)
    save(fig, 'maps.png')


def matchups():
    """Plotting.set_layout and Plotting.add_stats on the HYPERNETS / Sentinel-2 matchups."""
    stats = xr.open_dataset(MATCHUPS)
    good = stats.where(stats.good, drop=True)
    # visible and red-edge bands, where the water signal is (the near-infrared Rrs is close to 0)
    good = good.sel(wl=slice(0, 705))
    df = good[['Rrs_median', 'isRrs_nadir_median']].to_dataframe().dropna().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sc = ax.scatter(df.isRrs_nadir_median, df.Rrs_median, c=df.wl, cmap='Spectral_r', s=12)
    Plotting.set_layout(ax)
    Plotting.add_stats(df.isRrs_nadir_median.values, df.Rrs_median.values, ax, label=True, fontsize=10)
    ax.set(xlabel=r'in situ $R_{rs}$ (sr$^{-1}$), HYPERNETS', ylabel=r'Sentinel-2 $R_{rs}$ (sr$^{-1}$), GRS',
           title=f'Berre lagoon, {good.sizes["time"]} matchups, 443-705 nm')
    fig.colorbar(sc, ax=ax, shrink=0.7, label='wavelength (nm)')
    save(fig, 'matchups.png')


if __name__ == '__main__':
    for make in (geometry, irradiance, remove_wavelengths, terrain, maps, matchups):
        make()
