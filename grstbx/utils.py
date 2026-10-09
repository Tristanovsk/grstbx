"""
Miscellaneous helpers: geometry/time utilities (SpatioTemp), in-situ data readers (Data),
solar irradiance (Irradiance), plotting (Plotting), DEM-based illumination (Dem)
and regridding (Reproj).

Heavy optional dependencies (matplotlib, scipy, scikit-learn, xesmf) are imported
inside the functions that need them to keep ``import grstbx`` fast.
"""

import numpy as np
import pandas as pd

from affine import Affine
import xarray as xr
import geopandas as gpd

import importlib_resources

__all__ = ['SpatioTemp', 'Data', 'Irradiance', 'Plotting', 'Dem', 'Reproj']

class SpatioTemp():
    """Geometry and time helpers."""

    def get_time(self, ds, key='start_date'):
        """Assign a scalar ``time`` coordinate to ``ds`` from its ``key`` attribute."""
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
        :param ellps: ellipsoid used for geodesic computation
        :return: wkt of the box centered on provided coordinates
        '''
        from math import sqrt, atan, degrees, pi
        import pyproj
        geod = pyproj.Geod(ellps=ellps)

        # distance from the centre to each corner (half diagonal)
        half_diag = sqrt(width ** 2 + height ** 2) / 2

        azimuth1 = atan(width / height)
        azimuth2 = atan(-width / height)
        azimuth3 = azimuth1 + pi  # first point + 180 degrees
        azimuth4 = azimuth2 + pi  # second point + 180 degrees

        (pt1_lon, pt2_lon, pt3_lon, pt4_lon), (pt1_lat, pt2_lat, pt3_lat, pt4_lat), _ = geod.fwd(
            [center_lon] * 4, [center_lat] * 4,
            [degrees(a) for a in (azimuth1, azimuth2, azimuth3, azimuth4)], [half_diag] * 4)

        wkt_poly = 'POLYGON (( %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f ))' % (
            pt1_lon, pt1_lat, pt2_lon, pt2_lat, pt3_lon, pt3_lat, pt4_lon, pt4_lat, pt1_lon, pt1_lat)
        return wkt_poly

    def transform_from_latlon(self, lat, lon):
        """Affine transform of a regular grid defined by 1d ``lat`` and ``lon`` arrays."""
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(self, shapes, coords, latitude='lat', longitude='lon',
                  fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xarray coordinates. This only works for 1d latitude and longitude
        arrays.

        Example
        -------
        1. read shapefile to geopandas.GeoDataFrame
        2. encode the different shapes as different numbers (0.0, 1.0, ...), np.nan elsewhere
        3. assign this to a new coord of the original xarray.DataArray

        >>> states = gpd.read_file(shp_dir)
        >>> shapes = zip(states.geometry, range(len(states)))
        >>> ds['states'] = SpatioTemp().rasterize(shapes, ds.coords, longitude='X', latitude='Y')

        :param shapes: iterable of (geometry, value) pairs
        :param coords: coordinates of the output grid (e.g. ``ds.coords``)
        :param latitude: name of the latitude coordinate
        :param longitude: name of the longitude coordinate
        :param fill: value of the pixels outside the shapes
        :param kwargs: passed to ``rasterio.features.rasterize``
        :return: xarray.DataArray with the values of the shapes, ``fill`` outside,
                 and the (latitude, longitude) coordinates
        """
        from rasterio import features

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
    """Readers and helpers for in-situ / spectral data."""

    def __init__(self):
        pass

    def format_df(self, df, vars, wl):
        """
        Rebuild a 3-level column MultiIndex (l0, l1, l2) where l2 is the numeric
        wavelength obtained by substituting each band name of ``vars`` by its ``wl``.
        """
        h1 = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
        h2 = df.columns.get_level_values(0).str.replace('_.*', '', regex=True) + '_' + df.columns.get_level_values(1)
        h3 = df.columns.get_level_values(0)
        for band, num in reversed(list(zip(vars, wl))):
            h3 = h3.str.replace(band, str(num))
        h3 = pd.to_numeric(h3, errors='coerce')

        tuples = list(zip(h1, h2, h3))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        return df

    def read_aeronet_ocv3(self, file, skiprows=8):
        '''
        Read and format in pandas data.frame the standard AERONET-OC data

        :param file: AERONET-OC (version 3) csv file
        :param skiprows: number of header lines before the data
        :return: pandas.DataFrame indexed by date, with 3-level columns (name, data type, wavelength)
        '''
        self.file = file
        ifile = self.file

        h1 = pd.read_csv(ifile, skiprows=skiprows - 1, nrows=1).columns[3:]
        h1 = pd.Index(np.insert(np.asarray(h1, dtype=object), 0, 'site'))
        data_type = h1.str.replace(r'\[.*\]', '', regex=True)
        data_type = data_type.str.replace('Exact_Wave.*', 'wavelength', regex=True)
        # convert into float to order the dataframe with increasing wavelength
        h2 = h1.str.replace(r'.*\[', '', regex=True)
        h2 = h2.str.replace(r'nm\].*', '', regex=True)
        h2 = h2.str.replace(r'Exact_Wavelengths\(um\)_', '', regex=True)
        h2 = pd.to_numeric(h2, errors='coerce')
        h2 = h2.fillna('').T
        df = pd.read_csv(ifile, skiprows=skiprows, na_values=['N/A', -999.0, -9.999999], index_col=False)

        # columns 1 and 2 hold date and time
        date = pd.to_datetime(df.iloc[:, 1].astype(str) + ' ' + df.iloc[:, 2].astype(str),
                              format="%d:%m:%Y %H:%M:%S")
        df = df.drop(columns=df.columns[[1, 2]])
        df.index = pd.Index(date, name='date')

        tuples = list(zip(h1, data_type, h2))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in df.columns])
        df.sort_index(axis=1, level=2, inplace=True)
        return df

    @staticmethod
    def _wl_to_keep(wl, wl_to_remove):
        # only the wavelength coordinate is needed: no data is loaded or computed
        wl = np.asarray(wl)
        keep = np.ones(wl.shape, dtype=bool)
        for wl_min, wl_max in wl_to_remove:
            keep &= (wl < wl_min) | (wl > wl_max)
        return wl[keep]

    @staticmethod
    def remove_wl_dataarray(xarr, wl_to_remove, drop=True):
        """
        Remove wavelength ranges from a DataArray with a 'wl' dimension.

        :param xarr: xarray.DataArray
        :param wl_to_remove: list of (wl_min, wl_max) inclusive ranges to remove
        :param drop: kept for backward compatibility (unused)
        :return: xarray.DataArray without the removed wavelengths
        """
        return xarr.sel(wl=Data._wl_to_keep(xarr.wl, wl_to_remove))

    @staticmethod
    def remove_wl_dataset(xds, wl_to_remove, variable='Rrs', drop=True):
        """
        Remove wavelength ranges from a Dataset with a 'wl' dimension.

        :param xds: xarray.Dataset
        :param wl_to_remove: list of (wl_min, wl_max) inclusive ranges to remove
        :param variable: kept for backward compatibility (unused)
        :param drop: kept for backward compatibility (unused)
        :return: xarray.Dataset without the removed wavelengths
        """
        return xds.sel(wl=Data._wl_to_keep(xds.wl, wl_to_remove))


class Irradiance:
    """Extraterrestrial solar spectral irradiance (Thuillier et al., 2003 by default)."""

    def __init__(self, F0_file=importlib_resources.files('grstbx.data').joinpath('Thuillier_2003_0.3nm.dat')):
        self.F0_file = F0_file

    def load_F0(self, ):
        """Load the irradiance table into ``self.F0df`` (columns: wl, F0)."""
        self.F0df = pd.read_csv(self.F0_file, skiprows=15, sep='\t', header=None, names=('wl', 'F0'))

    def get_F0(self, wl, mute=False):
        '''
        interpolate and return solar spectral irradiance (mW/m2/nm)

        :param wl: wavelength in nm, scalar or np.array
        :param mute: if true values are not returned (only saved in object)
        :return:
        '''
        from scipy.interpolate import interp1d

        self.wl = wl
        self.F0 = interp1d(self.F0df.wl, self.F0df.F0, fill_value='extrapolate')(wl)

        if not mute:
            return self.F0


class Plotting:
    """Matplotlib helpers for multi-temporal images and scatter plots."""

    def __init__(self):
        pass

    def plot_wrap(self, data, ofig=None, title='', cmap='viridis', **kwargs):
        """
        Plot a (time, lat, lon) DataArray as a grid of maps (4 per row) with a shared colorbar.

        :param data: xarray.DataArray with 'time', 'lat' and 'lon' coordinates
        :param ofig: output figure file (multi-date only), not saved if None
        :param title: figure title
        :param cmap: matplotlib colormap
        :param kwargs: passed to xarray plot
        :return: matplotlib or FacetGrid plot object
        """

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

            if ofig is not None:
                p.fig.savefig(ofig)
        return p

    def _plot_image(self, data, factor=2.5, vmax=1, title=None, cmap=None, filename=None):
        """Plot each time slice of ``data`` (time, y, x[, rgb]) multiplied by ``factor`` on a 5-column grid."""
        import matplotlib.pyplot as plt

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
                im = ax.imshow(data[index] * factor, cmap=cmap, vmin=0.0, vmax=vmax, interpolation='nearest')
                ax.text(0, -2, caption, fontsize=12)
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
        """Square axes with identical x/y limits and a 1:1 line."""
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
        """
        Draw the linear regression of ``y`` vs ``x`` on ``ax`` and optionally
        print slope/intercept, r, rmse, mape and N.
        """
        from scipy import stats as sp_stats
        from sklearn import metrics

        regr = sp_stats.linregress(x, y)

        # Prediction metrics
        rmse = np.sqrt(metrics.mean_squared_error(x, y))
        mape = metrics.mean_absolute_percentage_error(x, y)
        bias = np.mean(y - x)
        N = len(x)
        stats = '$y = %.2f x %+.4f$' % (regr.slope, regr.intercept) + \
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
    """Terrain illumination from a Digital Elevation Model."""

    @staticmethod
    def compute_dem_attributes(dem_raster,
                               sza,
                               azi,
                               z_factor=1):
        '''
        Compute terrain slope, aspect and solar illumination (cosine of the local incidence angle):

        cos(theta_i) = cos(sza) cos(slope) + sin(sza) sin(slope) cos(azi - aspect)

        The gradients are computed with the x and y coordinates of the raster, which must be
        in meter (projected coordinate system).

        :param dem_raster: rioxarray Dataarray-like elevation in meter, dims (y, x)
        :param sza: solar zenith angle in degree
        :param azi: sun azimuth from North in degree (clockwise)
        :param z_factor: vertical exaggeration applied to the elevation (1: none)
        :return: xarray.Dataset with 'shaded' (cos(theta_i)), 'slope' and 'aspect'
                 (downslope direction, clockwise from North) variables, angles in radian
        '''

        # derivatives along the y (northing) and x (easting) coordinates
        dz_dy, dz_dx = np.gradient(z_factor * np.asarray(dem_raster, dtype=float),
                                   np.asarray(dem_raster.y, dtype=float),
                                   np.asarray(dem_raster.x, dtype=float))
        azir = np.radians(azi % 360)
        szar = np.radians(sza)

        slope = np.arctan(np.hypot(dz_dx, dz_dy))
        # azimuth of the downslope direction (-gradient), clockwise from North
        aspect = np.arctan2(-dz_dx, -dz_dy)

        shaded = np.cos(szar) * np.cos(slope) + np.sin(szar) * np.sin(slope) * np.cos(azir - aspect)

        return xr.Dataset(dict(shaded=(["y", "x"], shaded),
                               slope=(["y", "x"], slope),
                               aspect=(["y", "x"], aspect)),
                          coords=dict(x=dem_raster.x,
                                      y=dem_raster.y),
                          )


class Reproj():
    """Regridding of swath (curvilinear lat/lon) products onto a regular grid."""

    def __init__(self):
        pass

    @staticmethod
    def regridding(input_dataset,
                   output_grid_size=(1200, 1200),
                   d_input_crs=4326,
                   parallel=True,
                   latitude_name='latitude',
                   longitude_name='longitude',
                   method='bilinear',
                   chunk=500):
        """
        Take a PRISMA L1C product in sensor geometry (x,y) as input and
        return it in a georeferenced geometry (lon,lat).

        WARNING : Due to the use of the xESMF package, relying on Fortran,
        some user warnings like : "UserWarning: Input array is not F_CONTIGUOUS.
        Will affect performance." may be raised. It is not an issue in our case
        (see https://github.com/JiaweiZhuang/xESMF/issues/25).

        :param input_dataset: the product to regrid
        :param output_grid_size: (tuple) output grid size in (lon, lat) format
        :param d_input_crs: (int) code EPSG of the related geolocalisation frame
        :param parallel: compute regridding weights in parallel (dask)
        :param latitude_name: name of the latitude variable
        :param longitude_name: name of the longitude variable
        :param method: xESMF regridding method
        :param chunk: chunk size of the output grid

        :return output_dataset: the regularised product
        """
        import xesmf as xe

        # setting lon and lat as coordinates
        attrs = input_dataset.attrs

        # make the grid that the data will be regridded to
        grid_lons = np.linspace(input_dataset[longitude_name].min().values, input_dataset[longitude_name].max().values, output_grid_size[0])
        grid_lats = np.linspace(input_dataset[latitude_name].min().values, input_dataset[latitude_name].max().values, output_grid_size[1])
        new_grid = xr.Dataset({latitude_name: ([latitude_name], grid_lats), longitude_name: ([longitude_name], grid_lons)})
        new_grid = new_grid.chunk({latitude_name: chunk, longitude_name: chunk})

        # use periodic=False if either or both the lat and lon dimensions are not regular
        regridder = xe.Regridder(input_dataset, new_grid,
                                 method=method,
                                 periodic=False,
                                 unmapped_to_nan=True,
                                 parallel=parallel)

        # regrid the data
        output_dataset = regridder(input_dataset)

        # put "x","y" naming:
        output_dataset = output_dataset.rename({longitude_name: "x", latitude_name: "y"})

        # adding the CRS
        output_dataset.rio.write_crs(d_input_crs, inplace=True)
        output_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        output_dataset.rio.write_coordinate_system(inplace=True)
        output_dataset.attrs.update(attrs)
        return output_dataset