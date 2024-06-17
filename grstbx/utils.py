import numpy as np
import pandas as pd

from rasterio import features
from affine import Affine
import xarray as xr
import geopandas as gpd

import matplotlib.pyplot as plt
import importlib_resources
from scipy.interpolate import interp1d
import scipy as sp
from sklearn import linear_model, metrics


class SpatioTemp():

    def get_time(self, ds, key='start_date'):
        if key in ds.attrs.keys():
            grid_time = pd.to_datetime(ds.attrs[key])
            return ds.assign(time=grid_time)
        raise ValueError("Time attribute missing: {0}".format(key))

    def wktbox(self, center_lon, center_lat, width=100, height=100, ellps='WGS84'):
        '''

        :param center_lon: decimal longitude
        :param center_lat: decimal latitude
        :param width: width of the box in m
        :param height: height of the box in m
        :return: wkt of the box centered on provided coordinates
        '''
        from math import sqrt, atan, pi
        import pyproj
        geod = pyproj.Geod(ellps=ellps)

        rect_diag = sqrt(width ** 2 + height ** 2)

        azimuth1 = atan(width / height)
        azimuth2 = atan(-width / height)
        azimuth3 = atan(width / height) + pi  # first point + 180 degrees
        azimuth4 = atan(-width / height) + pi  # second point + 180 degrees

        pt1_lon, pt1_lat, _ = geod.fwd(center_lon, center_lat, azimuth1 * 180 / pi, rect_diag)
        pt2_lon, pt2_lat, _ = geod.fwd(center_lon, center_lat, azimuth2 * 180 / pi, rect_diag)
        pt3_lon, pt3_lat, _ = geod.fwd(center_lon, center_lat, azimuth3 * 180 / pi, rect_diag)
        pt4_lon, pt4_lat, _ = geod.fwd(center_lon, center_lat, azimuth4 * 180 / pi, rect_diag)

        wkt_point = 'POINT (%.6f %.6f)' % (center_lon, center_lat)
        wkt_poly = 'POLYGON (( %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f ))' % (
            pt1_lon, pt1_lat, pt2_lon, pt2_lat, pt3_lon, pt3_lat, pt4_lon, pt4_lat, pt1_lon, pt1_lat)
        return wkt_poly

    def transform_from_latlon(self, lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(self, shapes, coords, latitude='lat', longitude='lon',
                  fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xray coordinates. This only works for 1d latitude and longitude
        arrays.

        usage:
        -----
        1. read shapefile to geopandas.GeoDataFrame
              `states = gpd.read_file(shp_dir)`
        2. encode the different shapefiles that capture those lat-lons as different
            numbers i.e. 0.0, 1.0 ... and otherwise np.nan
              `shapes = (zip(states.geometry, range(len(states))))`
        3. Assign this to a new coord in your original xarray.DataArray
              `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

        arguments:
        ---------
        : **kwargs (dict): passed to `rasterio.rasterize` function

        attrs:
        -----
        :transform (affine.Affine): how to translate from latlon to ...?
        :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
          outside the .shp file with np.nan
        :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
          with "X", "Y" as keys, and xr.DataArray as values

        returns:
        -------
        :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
          and coords `Y` = latitude, 'X' = longitude.


        """
        # transform = transform_from_latlon(coords[latitude], coords[longitude])
        out_shape = (len(coords[latitude]), len(coords[longitude]))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill,  # transform=transform,
                                    dtype=float, **kwargs)
        spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
        return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

    @staticmethod
    def clip_raster(raster, lat, lon, extent_m):
        '''
        :param raster as rioxarray object with documented coordinate system
        :param lat: latitude (float) of central point of buffer
        :param lon: longitude (float) of central point of buffer
        :param extent_m: extent of square centered on lat/lon
        :param crs: crs for reprojection, use lat/lon epsg4326 otherwise.
        :return: clipped raster
        '''
        # distance is d/2 of the square buffer around the point,
        # from center to corner;
        # find buffer width in meters
        buffer_width_m = extent_m / np.sqrt(2)

        # EPSG:4326 sets Coordinate Reference System to WGS84 to match input
        wgs84_pt_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat], crs='4326'))

        # find suitable projected coordinate system for distance
        utm_crs = wgs84_pt_gdf.estimate_utm_crs()
        # reproject to UTM -> create square buffer (cap_style = 3) around point -> reproject back to WGS84
        buffer = wgs84_pt_gdf.to_crs(utm_crs).buffer(buffer_width_m, cap_style=3)
        # get buffer in the raster coordinate system
        buffer = buffer.to_crs(raster.rio.crs)

        # clipping
        return raster.rio.clip(buffer)


class Data:
    def __init__(self):
        pass

    def format_df(self, df, vars, wl):
        h1 = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
        h2 = df.columns.get_level_values(0).str.replace('_.*', '') + '_' + df.columns.get_level_values(1)
        h3 = df.columns.get_level_values(0)
        for band, num in reversed(list(zip(vars, wl))):
            h3 = h3.str.replace(band, str(num))
        h3 = pd.to_numeric(h3, errors='coerce')

        tuples = list(zip(h1, h2, h3))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        return df

    def read_aeronet_ocv3(self, file, skiprows=8):
        ''' Read and format in pandas data.frame the standard AERONET-OC data '''
        self.file = file
        dateparse = lambda x: pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
        ifile = self.file

        h1 = pd.read_csv(ifile, skiprows=skiprows - 1, nrows=1).columns[3:]
        h1 = np.insert(h1, 0, 'site')
        data_type = h1.str.replace('\[.*\]', '')
        data_type = data_type.str.replace('Exact_Wave.*', 'wavelength')
        # convert into float to order the dataframe with increasing wavelength
        h2 = h1.str.replace('.*\[', '')
        h2 = h2.str.replace('nm\].*', '')
        h2 = h2.str.replace('Exact_Wavelengths\(um\)_', '')
        h2 = pd.to_numeric(h2, errors='coerce')  # h2.str.extract('(\d+)').astype('float')
        h2 = h2.fillna('').T
        df = pd.read_csv(ifile, skiprows=skiprows, na_values=['N/A', -999.0, -9.999999], parse_dates={'date': [1, 2]},
                         date_parser=dateparse, index_col=False)

        # df['site'] = site
        # df.set_index(['site', 'date'],inplace=True)
        df.set_index('date', inplace=True)

        tuples = list(zip(h1, data_type, h2))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in df.columns])
        df.sort_index(axis=1, level=2, inplace=True)
        return df


class Irradiance:
    def __init__(self, F0_file=importlib_resources.files('grstbx.data').joinpath('Thuillier_2003_0.3nm.dat')):
        self.F0_file = F0_file

    def load_F0(self, ):
        self.F0df = pd.read_csv(self.F0_file, skiprows=15, sep='\t', header=None, names=('wl', 'F0'))

    def get_F0(self, wl, mute=False):
        '''
        interpolate and return solar spectral irradiance (mW/m2/nm)

        :param wl: wavelength in nm, scalar or np.array
        :param mute: if true values are not returned (only saved in object)
        :return:
        '''
        self.wl = wl
        self.F0 = interp1d(self.F0df.wl, self.F0df.F0, fill_value='extrapolate')(wl)

        if not mute:
            return self.F0


class Plotting:
    def __init__(self):
        pass

    def plot_wrap(self, data, ofig=None, title='', cmap='viridis', **kwargs):

        # fig, (ax, cax) = plt.subplots(nrows=2, figsize=(15, 35),
        #                               gridspec_kw={"height_ratios": [1, 0.05]})

        nimg = data.time.shape[0]
        nrows = nimg // 4 + (1 if nimg % 4 else 0)
        if nimg == 1:
            p = data.isel().plot(x='lon', y='lat', robust=True, size=10, cmap=cmap,
                                 cbar_kwargs=dict(orientation='horizontal', pad=.1, aspect=40, shrink=0.6), **kwargs)
        else:
            p = data.isel().plot(x='lon', y='lat', col='time', col_wrap=min(nimg, 4), robust=True, size=10, cmap=cmap,
                                 vmin=0, cbar_kwargs=dict(orientation='horizontal', pad=.1, aspect=40, shrink=0.6),
                                 **kwargs)
            for i, ax in enumerate(p.axes.flat):
                if i >= data.time.shape[0]:
                    break
                ax.set_title(pd.to_datetime(data.time[i].values))
            p.cbar.remove()
            p.fig.set_size_inches(15, nrows * 6)
            p.fig.suptitle(title)
            p.fig.subplots_adjust(bottom=0.1, top=0.9, left=0.08, right=0.92, wspace=0.02, hspace=0.2)
            p.add_colorbar(orientation='horizontal', pad=.1, aspect=40, shrink=0.6, anchor=(0, 1))

            if ofig != None:
                p.fig.savefig(ofig)
        return p

    def _plot_image(self, data, factor=2.5, vmax=1, title=None, cmap=None, filename=None):
        rows = data.shape[0] // 5 + (1 if data.shape[0] % 5 else 0)
        aspect_ratio = (1.0 * data.shape[1]) / data.shape[2]
        fig, axs = plt.subplots(nrows=rows, ncols=5, figsize=(15, 3 * rows * aspect_ratio))
        for index, ax in enumerate(axs.flatten()):
            if index < data.shape[0] and index < len(data.time):
                time = pd.to_datetime(data.time[index].values)
                caption = str(index) + ': ' + time.strftime('%Y-%m-%d')
                # if self.cloud_coverage is not None:
                #     caption = caption + '(' + "{0:2.0f}".format(self.cloud_coverage[index] * 100.0) + '%)'

                ax.set_axis_off()
                im = ax.imshow(data[index] * factor if data[index].shape[-1] == 3 or data[index].shape[-1] == 4 else
                               data[index] * factor, cmap=cmap, vmin=0.0, vmax=vmax, interpolation='nearest')
                ax.text(0, -2, caption, fontsize=12)  # , color='r') if self.mask[index] else 'g')
            else:
                ax.set_axis_off()
        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.85,
                            wspace=0.02, hspace=0.2)
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)

        fig.suptitle(title, fontsize=18)

        if filename:
            plt.savefig(filename)  # , bbox_inches='tight')
            plt.close()

    @staticmethod
    def set_layout(ax):
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        return ax

    @staticmethod
    def add_stats(x, y, ax, label=False, fontsize=12):
        regr = sp.stats.linregress(x, y)

        # Prediction metrics
        rmse = np.sqrt(metrics.mean_squared_error(x, y))
        mape = metrics.mean_absolute_percentage_error(x, y)
        bias = np.mean(y - x)
        N = len(x)
        stats = f'$y = %.2f x %+.4f$' % (regr.slope, regr.intercept) + \
                '\n$r = %.3f$' % (regr.rvalue) + \
                '\n$rmse = %.3f$' % (rmse) + \
                '\n$mape = %.3f$' % (mape) + \
                '\n$N = %i$' % (N)
        ax.axline(xy1=(0.01, regr.intercept + 0.01 * regr.slope), slope=regr.slope, ls='--', lw=1.5, c="gray")
        if label:
            ax.text(0.98, 0.01, stats, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes)
        return


class Dem:

    @staticmethod
    def compute_dem_attributes(dem_raster,
                               sza,
                               azi):
        '''

        :param dem_raster: rioxarray Dataarray-like elevation in meter
        :param sza: solar zenith angle in degree
        :param azi: sun azimuth from North in degree
        :return:
        '''

        x, y = np.gradient(dem_raster)
        azir = np.radians(azi % 360)
        szar = np.radians(sza)

        slope = np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)

        shaded = np.cos(szar) * np.cos(slope) + np.sin(szar) * np.sin(slope) * np.cos(azir - aspect)

        return xr.Dataset(dict(shaded=(["y", "x"], shaded),
                               slope=(["y", "x"], slope)),
                          coords=dict(x=dem_raster.x,
                                      y=dem_raster.y),
                          )
