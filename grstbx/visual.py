"""
Interactive (holoviews / panel / bokeh) viewers for GRS L2A and L2B datacubes,
intended to be used in Jupyter notebooks.

 - ViewSpectral: browse L2A images (true-colour composite or bands) by date, draw areas / points
   of interest and plot the spectra of the points
 - ViewParam: browse L2B parameters by date
 - ImageViewer: older viewers with on-the-fly spectrum extraction

ViewSpectral and ViewParam read only the pixels visible on the screen, at the resolution of the
screen: from the pyramids of the GRS Zarr products, or from coarser levels computed on demand for the
other inputs (see Multiscale).
"""

import os
from collections import OrderedDict
from collections import OrderedDict as odict

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401, activates the .rio accessor
import pyproj
import geopandas as gpd

import holoviews as hv
from holoviews.element import tiles as hvts
from holoviews import opts
from holoviews.plotting.links import DataLink

import bokeh
import colorcet as cc
import panel as pn
import param as pm
from shapely.geometry import Polygon

hv.extension('bokeh')

# Free basemaps (XYZ tiles, web Mercator) proposed in the ViewSpectral / ViewParam widgets.
# The Stamen styles are now served by Stadia Maps: free without key when the notebook runs on
# localhost, otherwise a (free) API key is needed, see set_stadia_api_key.
BASEMAPS = {
    'CARTO Positron': 'https://a.basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png',
    'CARTO Positron (no labels)': 'https://a.basemaps.cartocdn.com/light_nolabels/{Z}/{X}/{Y}.png',
    'CARTO Dark Matter': 'https://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png',
    'CARTO Voyager': 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png',
    'OpenStreetMap': 'https://tile.openstreetmap.org/{Z}/{X}/{Y}.png',
    'OpenTopoMap': 'https://a.tile.opentopomap.org/{Z}/{X}/{Y}.png',
    'Esri World Imagery': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}',
    'Esri World Topo': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{Z}/{Y}/{X}',
    'Esri Ocean': 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{Z}/{Y}/{X}',
    'Stamen Toner (Stadia)': 'https://tiles.stadiamaps.com/tiles/stamen_toner/{Z}/{X}/{Y}.png',
    'Stamen Toner Lite (Stadia)': 'https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{Z}/{X}/{Y}.png',
    'Stamen Terrain (Stadia)': 'https://tiles.stadiamaps.com/tiles/stamen_terrain/{Z}/{X}/{Y}.png',
    'Stamen Watercolor (Stadia)': 'https://tiles.stadiamaps.com/tiles/stamen_watercolor/{Z}/{X}/{Y}.jpg',
    'Alidade Smooth (Stadia)': 'https://tiles.stadiamaps.com/tiles/alidade_smooth/{Z}/{X}/{Y}.png',
    'Alidade Smooth Dark (Stadia)': 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{Z}/{X}/{Y}.png',
}
DEFAULT_BASEMAP = 'Esri World Imagery'
_STADIA_API_KEY = os.environ.get('STADIA_API_KEY')


def set_stadia_api_key(api_key):
    """
    Set the Stadia Maps API key used by the Stamen / Alidade basemaps (free account on
    https://stadiamaps.com); not needed when the notebook runs on localhost. The key can also be
    given by the ``STADIA_API_KEY`` environment variable.
    """
    global _STADIA_API_KEY
    _STADIA_API_KEY = api_key


def basemap_tiles(name):
    """
    holoviews Tiles of a basemap of :data:`BASEMAPS` (or any XYZ URL with {X}, {Y}, {Z}).

    :param name: name in :data:`BASEMAPS`, or XYZ URL
    :return: hv.Tiles
    """
    url = BASEMAPS.get(name, name)
    if 'stadiamaps.com' in url and _STADIA_API_KEY:
        url += '?api_key=' + _STADIA_API_KEY
    return hv.Tiles(url, name=name)


# colormaps proposed in the ViewSpectral / ViewParam widgets
COLORMAPS = ['CET_D13', 'bky', 'CET_D1A', 'CET_CBL2', 'CET_L10', 'CET_C6s',
             'kbc', 'blues_r', 'kb', 'rainbow', 'fire', 'kgy', 'bjy', 'gray']


class ImageViewer():
    """Older interactive viewers built on param.Parameterized."""

    def Rrs_date(self, raster, third_dim='wl', param='Rrs', Rrs_unit=True):
        """
        Map of one band / one date with a box-drawing tool; the mean (+/- std)
        spectrum within each drawn box is plotted alongside.

        :param raster: Dataset with dims (time, wl, y, x), in EPSG:3857 to overlay basemaps
        :param third_dim: spectral dimension
        :param param: variable to display
        :param Rrs_unit: if False, values are displayed as rho_w = pi * Rrs
        :return: panel layout
        """

        param_label = r'$R_{rs}$'
        if not Rrs_unit:
            param_label = r'$rho_w$'

        ps = odict([(n, cc.palette[n]) for n in
                    ['gouldian', 'rainbow', 'fire', 'CET_D13', 'CET_CBC1', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc']])

        iwls = {wl: iwl for iwl, wl in enumerate(raster.wl.values)}

        dates = {str(date): idate for idate, date in enumerate(raster.time.values)}

        # map_tiles = EsriImagery().opts(alpha=0.65, bgcolor='black')
        maps = ['StamenTonerBackground', 'EsriImagery', 'EsriUSATopo', 'EsriTerrain', 'StamenWatercolor']
        bases = odict(
            [(name, None) if name == 'None' else (name, getattr(hvts, name)().relabel(name)) for name in maps])
        gopts = hv.opts.Tiles(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

        polys = hv.Polygons([])
        # poly_stream = hv.streams.PolyDraw(source=polys, drag=True, show_vertices=True)
        # poly_edit = hv.streams.PolyEdit(source=polys, shared=True)
        box_stream = hv.streams.BoxEdit(source=polys)
        # pointer = hv.streams.PointerXY(source=im)
        opts.defaults(
            opts.Curve(tools=['hover'], shared_axes=False, framewise=True), )

        class Viewer(pm.Parameterized):

            date = pm.Selector(dates)
            wavelength = pm.Selector(iwls, default=2)
            cmap = pm.Selector(ps, default=ps['gouldian'])
            basemap = pm.Selector(bases)
            vmax = pm.Number(0.03)
            extracted_data = []

            @pm.depends('date')
            def extract_ds_by_date(self):
                self.ds_ = hv.Dataset(raster.isel(time=self.date, drop=True).compute())
                # clean up graph

            @pm.depends('date')
            def clean_up(self):
                return hv.NdOverlay({0: hv.Curve([], 'Wavelength (nm)', param_label)})

            # @pn.depends(box_stream)
            @pm.depends('date')
            def roi_curves(self, data):

                # if no data selected: plot empty graph
                if not data or not any(len(d) for d in data.values()):
                    return hv.NdOverlay({0: hv.Curve([], 'Wavelength (nm)', param_label)})
                ds_ = self.ds_  # hv.Dataset(raster.isel(time=self.date,drop=True))
                curves = {}
                data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
                i = 0
                for x0, x1, y0, y1 in data:
                    if i == 0:
                        self.extracted_data = raster.sel(x=slice(x0, x1), y=slice(y0, y1))
                    selection = ds_.select(x=(x0, x1), y=(y0, y1))

                    mean = selection.aggregate(third_dim, np.nanmean).data
                    if np.isnan(mean[param][0]):
                        continue

                    if not Rrs_unit:
                        mean = mean * np.pi

                    wl = mean.wl
                    curves[i] = hv.Curve((wl, mean[param]), 'Wavelength (nm)',
                                         param_label)  # * hv.Spread((wl,mean[param],std[param])).opts(fill_alpha=0.3)
                    i += 1

                if i > 0:
                    return hv.NdOverlay(curves)
                else:
                    return hv.NdOverlay({1: hv.Curve([], 'Wavelength (nm)', param_label)})

            # a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
            # @pm.depends('date')
            def add_envelope(self, data={}):
                if not data or not any(len(d) for d in data.values()):
                    return hv.NdOverlay({0: hv.Curve([], 'Wavelength (nm)', param_label)})
                d_ = self.ds_  # hv.Dataset(raster.isel(time=self.date,drop=True))
                envelope = {}
                data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
                i = 0
                for x0, x1, y0, y1 in data:

                    selection = d_.select(x=(x0, x1), y=(y0, y1))
                    mean = selection.aggregate(third_dim, np.nanmean).data
                    if np.isnan(mean[param][0]):
                        continue
                    std = selection.aggregate(third_dim, np.nanstd).data
                    if not Rrs_unit:
                        mean = mean * np.pi
                        std = std * np.pi
                    wl = mean.wl
                    envelope[i] = hv.Spread((wl, mean[param], std[param]), fill_alpha=0.3)
                    i += 1
                if i > 0:
                    return hv.NdOverlay(envelope)
                else:
                    return hv.NdOverlay({1: hv.Curve([], 'Wavelength (nm)', param_label)})

            @pm.depends('wavelength')
            def add_line(self):
                return hv.NdOverlay(hv.VLine(self.wavelength))

            @pm.depends('wavelength', 'date', 'cmap')
            def select_band(self):
                d_ = raster.isel(wl=self.wavelength, time=self.date)
                title = 'Band at {:.2f}'.format(d_.wl.values) + ' nm'
                ds = hv.Dataset(d_)
                title = str(d_.time.dt.strftime("%Y/%m/%d, %H:%M:%S").values)

                im = ds.to(hv.Image, ['x', 'y']).opts(clim_percentile=True, padding=0, active_tools=['box_edit'],
                                                      tools=['hover', 'lasso_select'], title=title, cmap=self.cmap,
                                                      colorbar=True, clim=(0, self.vmax)).opts(
                    fontsize={'title': 18, 'labels': 14, 'xticks': 12, 'yticks': 12})  # .hist(bin_range=(0,0.02) )

                return (im * polys).opts(opts.Polygons(fill_alpha=0.2, line_color='black'))

            @pm.depends('basemap')
            def tiles(self):
                if self.basemap is None:
                    return
                return self.basemap.opts(gopts).opts(alpha=0.5)

            def map_band(self):
                self.extract_ds_by_date()
                return hv.DynamicMap(self.tiles) * self.select_band()  # .opts(**ropts)

        viewer = Viewer()
        # spectrum = hv.DynamicMap(viewer.roi_curves,streams=[pointer])
        mean = hv.DynamicMap(viewer.roi_curves, streams=[box_stream])
        std = hv.DynamicMap(viewer.add_envelope, streams=[box_stream])
        cleanup = hv.DynamicMap(viewer.clean_up)
        hlines = (viewer.add_line)
        graph = (mean * std * cleanup).opts(opts.Curve(height=500, width=700, framewise=True, xlim=(400, 1000)),
                                            opts.Polygons(fill_alpha=0.2, color='green', active_tools=['poly_draw']),
                                            # ,line_color='black'),
                                            opts.VLine(color='black')).opts(
            align='center', title='Extracted values (mean +/- std)',
            fontsize={'title': 18, 'labels': 14, 'xticks': 12, 'yticks': 12})
        # show_coords = viewer.show_poly_coords
        return pn.Column(pn.Param(viewer.param, default_layout=pn.Row, sizing_mode='stretch_width'),
                         pn.Row(viewer.map_band, graph))

    def param_date(self, raster, cmap='kbc'):
        """
        Map of a single-variable DataArray (time, y, x) with date / colormap / basemap selectors.

        :param raster: DataArray in EPSG:3857 to overlay basemaps
        :param cmap: initial colormap
        :return: panel layout
        """

        cmap_ = cmap
        ps = odict([(n, cc.palette[n]) for n in ['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc']])
        dates = {str(date): idate for idate, date in enumerate(raster.time.values)}
        maps = ['EsriImagery', 'EsriUSATopo', 'EsriTerrain', 'StamenWatercolor', 'StamenTonerBackground']
        bases = odict(
            [(name, None) if name == 'None' else (name, getattr(hvts, name)().relabel(name)) for name in maps])
        gopts = hv.opts.Tiles(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

        import holoviews.operation.datashader as hd

        class Viewer(pm.Parameterized):
            cmap = pm.Selector(ps, default=ps[cmap_])
            date = pm.Selector(dates)
            basemap = pm.Selector(bases)

            @pm.depends('date', 'cmap')
            def select_date(self):
                d_ = raster.isel(time=self.date)

                hv_dataset_large = hv.Dataset(d_, kdims=['x', 'y'])
                hv_image_large = hv.Image(hv_dataset_large, ['x', 'y']).opts(width=900, height=600)

                return hd.regrid(hv_image_large).opts(tools=['hover'], active_tools=['wheel_zoom'], cmap=self.cmap,
                                                      colorbar=True, colorbar_position='bottom',
                                                      clim=(0, None), cnorm='eq_hist')  # )

            @pm.depends('basemap')
            def tiles(self):
                if self.basemap is None:
                    return
                return self.basemap.opts(gopts).opts(alpha=0.5)

            def map_band(self):
                return hv.DynamicMap(self.tiles) * self.select_date()

        viewer = Viewer()

        return pn.Row(pn.Param(viewer.param), viewer.map_band)


class Utils():
    """Helpers shared by the viewers to retrieve user-drawn geometries."""

    @staticmethod
    def get_points(poi_stream,
                   crs=4326,
                   src_crs=3857):
        """
        Return the points drawn with a PointDraw stream as a GeoDataFrame.

        :param poi_stream: holoviews PointDraw stream (e.g. ``ViewSpectral.poi_stream``)
        :param crs: output CRS
        :param src_crs: CRS of the drawn coordinates (CRS of the map)
        :return: geopandas.GeoDataFrame of points (with their 'color' column if present)
        """
        geom = poi_stream.data
        points = gpd.GeoDataFrame(
            {k: v for k, v in geom.items() if k not in ('x', 'y')},
            geometry=gpd.points_from_xy(geom['x'], geom['y']), crs=src_crs)
        return points.to_crs(crs)

    @staticmethod
    def get_geom(aoi_stream,
                 crs=4326,
                 index=-1,
                 src_crs=3857):
        """
        Return one polygon drawn with a PolyDraw stream as a GeoDataFrame.

        :param aoi_stream: holoviews PolyDraw stream (e.g. ``ViewSpectral.aoi_stream``)
        :param crs: output CRS
        :param index: index of the polygon to return (default: last drawn)
        :param src_crs: CRS of the drawn coordinates (CRS of the map)
        :return: geopandas.GeoDataFrame with a single polygon
        """
        geom = aoi_stream.data
        ys, xs = geom['ys'][index], geom['xs'][index]
        polygon_geom = Polygon(zip(xs, ys))
        polygon = gpd.GeoDataFrame(index=[0], crs=src_crs, geometry=[polygon_geom])
        return polygon.to_crs(crs)

    @staticmethod
    def custom_hover(field='image'):
        """Bokeh hover tool displaying lon/lat (converted from web mercator) and pixel value."""
        formatter_code = """
          var digits = 4;
          var projections = Bokeh.require("core/util/projections");
          var x = special_vars.x; var y = special_vars.y;
          var coords = projections.wgs84_mercator.invert(x, y);
          return "" + (Math.round(coords[%d] * 10**digits) / 10**digits).toFixed(digits)+ "";
        """
        formatter_code_x, formatter_code_y = formatter_code % 0, formatter_code % 1
        custom_tooltips = [('Lon', '@x{custom}'), ('Lat', '@y{custom}'), ('Value', '@' + field + '{0.0000}')]
        custom_formatters = {
            '@x': bokeh.models.CustomJSHover(code=formatter_code_x),
            '@y': bokeh.models.CustomJSHover(code=formatter_code_y)
        }
        return bokeh.models.HoverTool(tooltips=custom_tooltips, formatters=custom_formatters)


def _coarsen2(da):
    """Halve the resolution of an image: 2x2 mean, or nearest neighbour for integer (flag) layers."""
    if da.dtype.kind in 'iub':
        return da.isel(x=slice(None, None, 2), y=slice(None, None, 2))
    return da.coarsen(x=2, y=2, boundary='trim').mean()


class Multiscale():
    """
    Lazy multiscale access to the 2-D layers (bands or variables) of a series of images, used by the
    viewers to read only the pixels displayed on the screen.

    Each image is a list of resolution levels (0: full resolution, 1: 2x coarser, ...). The levels of
    Zarr pyramids are read from the store (see :func:`grstbx.driver.open_zarr_image`); for the other
    inputs (and single-resolution stores), the coarser levels are computed on demand by 2x2 averaging
    (nearest neighbour for integer variables) of the layer displayed, and kept in memory.

    Example
    -------
    >>> ms = Multiscale.from_zarr(['S2A_L2A_1.zarr', 'S2B_L2A_2.zarr'])
    >>> ms = Multiscale.from_xarray(datacube.Rrs)

    :param images: list (one item per image) of lists of resolution levels (xarray objects with
                   x, y dimensions and a CRS), full resolution first
    :param times: acquisition times of the images
    :param min_size: size (pixels) of the longest side of the coarsest computed level
    """

    def __init__(self, images, times, min_size=512):
        self.images = images
        self.times = np.asarray(times)
        self.min_size = min_size
        self.crs = images[0][0].rio.crs
        self._cache = {}

    @classmethod
    def from_xarray(cls, obj, min_size=512):
        """
        :param obj: DataArray (e.g. ``Rrs`` with a ``wl`` dimension) or Dataset with x, y and optional
                    ``time`` dimensions
        """
        if 'time' not in obj.dims:
            obj = obj.expand_dims('time')
        images = [[obj.isel(time=itime)] for itime in range(obj.sizes['time'])]
        return cls(images, obj.time.values, min_size=min_size)

    @classmethod
    def from_zarr(cls, paths, min_size=512):
        """
        :param paths: path or list of paths of GRS Zarr stores (L2A or L2B), with or without pyramids
        """
        from .driver import open_zarr_image, zarr_levels

        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        images, times = [], []
        for path in paths:
            nlevels = max(len(zarr_levels(path)), 1)
            levels = [open_zarr_image(path, level=level) for level in range(nlevels)]
            time = levels[0].coords.get('time')
            if time is None:
                time = levels[0].attrs.get('acquisition_date', levels[0].attrs.get('start_date'))
            images.append(levels)
            times.append(np.datetime64(pd.Timestamp(np.asarray(time).ravel()[0]).tz_localize(None), 'ns'))
        order = np.argsort(times)
        return cls([images[i] for i in order], np.array(times)[order], min_size=min_size)

    @staticmethod
    def _select(obj, name):
        """Layer ``name`` of an image: variable of a Dataset, or wavelength(s) of Rrs."""
        if isinstance(obj, xr.Dataset):
            if not isinstance(name, (list, tuple)) and name in obj.data_vars:
                da = obj[name]
            else:
                da = obj['Rrs'].sel(wl=list(name) if isinstance(name, (list, tuple)) else name)
        elif 'wl' in obj.dims:
            da = obj.sel(wl=list(name) if isinstance(name, (list, tuple)) else name)
        else:
            da = obj
        return da.drop_vars([c for c in da.coords if c not in ('x', 'y', 'wl', 'spatial_ref')])

    def nlevels(self, idate):
        """Number of resolution levels of the image ``idate`` (stored or computed)."""
        levels = self.images[idate]
        if len(levels) > 1:
            return len(levels)
        size = max(levels[0].sizes['x'], levels[0].sizes['y'])
        return 1 + max(0, int(np.ceil(np.log2(size / self.min_size))))

    def resolution(self, idate, level=0):
        """Pixel size of a level (in units of the CRS)."""
        levels = self.images[idate]
        res = abs(float(levels[min(level, len(levels) - 1)].x[1] - levels[min(level, len(levels) - 1)].x[0]))
        return res * 2 ** max(0, level - len(levels) + 1)

    def bounds(self, idate=0):
        """(xmin, ymin, xmax, ymax) of the image ``idate`` in its CRS."""
        return self.images[idate][0].rio.bounds()

    def layer(self, idate, name, level=0):
        """Lazy (stored levels) or in-memory (computed levels) 2-D or 3-D layer of an image."""
        levels = self.images[idate]
        if level < len(levels):
            return self._select(levels[level], name)
        key = (idate, tuple(name) if isinstance(name, (list, tuple)) else name, level)
        if key not in self._cache:
            self._cache[key] = _coarsen2(self.layer(idate, name, level - 1)).load()
        return self._cache[key]

    def select_level(self, idate, target_res):
        """Coarsest level whose pixels are not larger than ``target_res``."""
        level = 0
        for candidate in range(1, self.nlevels(idate)):
            if self.resolution(idate, candidate) <= target_res:
                level = candidate
        return level

    def window(self, idate, name, bounds, target_res):
        """
        Read the pixels of a layer inside ``bounds`` at the coarsest level not coarser than ``target_res``.

        :param idate: index of the image
        :param name: variable name, wavelength or list of wavelengths
        :param bounds: (xmin, ymin, xmax, ymax) in the CRS of the image
        :param target_res: size of a screen pixel, in units of the CRS
        :return: (in-memory DataArray, level)
        """
        level = self.select_level(idate, target_res)
        da = self.layer(idate, name, level)
        res = self.resolution(idate, level)
        xmin, ymin, xmax, ymax = bounds
        yslice = slice(ymax + res, ymin - res) if da.y[0] > da.y[-1] else slice(ymin - res, ymax + res)
        return da.sel(x=slice(xmin - res, xmax + res), y=yslice).load(), level

    def pixel(self, idate, x, y):
        """Full-resolution values (all wavelengths for Rrs) of the pixel nearest to (x, y), or None outside."""
        obj = self.images[idate][0]
        da = obj['Rrs'] if isinstance(obj, xr.Dataset) and 'Rrs' in obj else obj
        xmin, ymin, xmax, ymax = self.bounds(idate)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return None
        return da.sel(x=x, y=y, method='nearest').load()


class _MultiscaleViewer(Utils):
    """
    Base of the viewers: map rendered from the pixels visible on the screen only.

    At each pan / zoom, the visible window is read at the coarsest resolution level that still has at
    least one image pixel per screen pixel (see :class:`Multiscale`) and, with ``reproject=True``,
    reprojected to web Mercator (EPSG:3857) to overlay basemaps. The last windows are kept in memory
    so that changing the colormap, the color range or the opacity does not read the data again.
    """

    def __init__(self, data, dates=None, reproject=False, minmaxvalues=(0, 0.02), minmax=(0, 0.06),
                 width=1200, height=800, min_size=512, basemap=DEFAULT_BASEMAP):
        if not isinstance(data, Multiscale):
            if isinstance(data, (str, os.PathLike)) or (
                    isinstance(data, (list, tuple)) and isinstance(data[0], (str, os.PathLike))):
                data = Multiscale.from_zarr(data, min_size=min_size)
            else:
                data = Multiscale.from_xarray(data, min_size=min_size)
        self.data = data

        # images to display
        self.indexes = list(range(len(data.times)))
        if dates is not None:
            dates = {pd.Timestamp(date).date() for date in dates}
            self.indexes = [i for i in self.indexes if pd.Timestamp(data.times[i]).date() in dates]
        self.datetimes = [pd.Timestamp(data.times[i]) for i in self.indexes]
        self.dates = np.array([dt.date() for dt in self.datetimes])

        # layout settings
        self.width, self.height = width, height
        self.minmaxvalues = minmaxvalues
        self.minmax = minmax
        self.colormaps = COLORMAPS

        # display CRS and transformations from / to the CRS of the images
        self.reproject = reproject
        self.basemap = basemap
        self.display_crs = pyproj.CRS.from_epsg(3857) if reproject else pyproj.CRS.from_user_input(data.crs)
        self._to_native = pyproj.Transformer.from_crs(self.display_crs, data.crs, always_xy=True)
        self._to_display = pyproj.Transformer.from_crs(data.crs, self.display_crs, always_xy=True)
        self._windows = OrderedDict()
        self._max_windows = 16

        # declare streaming object to get Area of Interest (AOI), in the CRS of the map
        self.aoi_polygons = hv.Polygons([]).opts(opts.Polygons(
            fill_alpha=0.3, fill_color='white', line_width=1.2))
        self.aoi_stream = hv.streams.PolyDraw(source=self.aoi_polygons, drag=True)
        self.edit_stream = hv.streams.PolyEdit(source=self.aoi_polygons, vertex_style={'color': 'red'})

    # -- geometries drawn on the map (in the CRS of the map)

    def get_geom(self, aoi_stream=None, crs=4326, index=-1):
        """Polygon drawn on the map (default: last one) as a GeoDataFrame in ``crs``."""
        return Utils.get_geom(aoi_stream or self.aoi_stream, crs=crs, index=index, src_crs=self.display_crs)

    def get_points(self, poi_stream=None, crs=4326):
        """Points drawn on the map as a GeoDataFrame in ``crs``."""
        return Utils.get_points(poi_stream or self.poi_stream, crs=crs, src_crs=self.display_crs)

    # -- rendering

    def _extent(self, idate=0):
        """Full extent of an image in the CRS of the map."""
        xmin, ymin, xmax, ymax = self._to_display.transform_bounds(*self.data.bounds(self.indexes[idate]))
        return (xmin, xmax), (ymin, ymax)

    def _read(self, idate, name, x_range, y_range, width, height):
        """Visible window of a layer, in the CRS of the map (cached)."""
        bounds = self._to_native.transform_bounds(x_range[0], y_range[0], x_range[1], y_range[1])
        target_res = max((bounds[2] - bounds[0]) / width, (bounds[3] - bounds[1]) / height)
        image = self.indexes[idate]
        level = self.data.select_level(image, target_res)
        key = (image, tuple(name) if isinstance(name, (list, tuple)) else name, level,
               tuple(np.round(bounds, 0)))
        if key in self._windows:
            self._windows.move_to_end(key)
            return self._windows[key]

        da, level = self.data.window(image, name, bounds, target_res)
        if da.sizes['x'] < 2 or da.sizes['y'] < 2:
            da = None
        elif self.reproject:
            da = (da.astype(np.float32).rio.write_crs(self.data.crs)
                  .rio.reproject(self.display_crs, nodata=np.nan))
        self._windows[key] = da
        if len(self._windows) > self._max_windows:
            self._windows.popitem(last=False)
        return da

    def _ranges(self, idate, x_range, y_range):
        if x_range is None or y_range is None:
            return self._extent(idate)
        return x_range, y_range

    def _image(self, idate, name, title, x_range=None, y_range=None, width=None, height=None):
        """hv.Image of the visible window of a layer."""
        x_range, y_range = self._ranges(idate, x_range, y_range)
        da = self._read(idate, name, x_range, y_range, width or self.width, height or self.height)
        if da is None:
            return hv.Image(np.full((1, 1), np.nan), bounds=(x_range[0], y_range[0], x_range[1], y_range[1]),
                            vdims=['image']).opts(title=title)
        return hv.Image(da.rename('image').squeeze(drop=True), kdims=['x', 'y'], vdims=['image']).opts(title=title)

    def _image_opts(self):
        hover = self.custom_hover('image') if self.reproject else 'hover'
        return dict(width=self.width, height=self.height, colorbar=True, tools=[hover],
                    active_tools=['wheel_zoom'], clipping_colors={'NaN': '#00000000'})

    def _tiles(self, basemap):
        gopts = hv.opts.Tiles(xaxis=None, yaxis=None, bgcolor='black', show_grid=False,
                              active_tools=['wheel_zoom'])
        return basemap_tiles(basemap).opts(gopts).opts(width=self.width, height=self.height)

    def _basemap_widget(self):
        # basemaps are in web Mercator: only available when the images are reprojected
        if not self.reproject:
            return pn.widgets.Select(value='None (reproject=False)', options=['None (reproject=False)'],
                                     disabled=True)
        options = list(BASEMAPS)
        if self.basemap not in options:
            # custom XYZ URL
            options.append(self.basemap)
        return pn.widgets.Select(value=self.basemap, options=options)

    def _date_widget(self):
        options = {str(dt): idate for idate, dt in enumerate(self.datetimes)}
        return pn.widgets.Select(value=0, options=options)

    def _map(self, layer_widget, date_widget, cmap_widget, opacity_widget, range_widget, basemap_widget,
             title_func):
        """Overlay of the basemap and of the multiscale image driven by the widgets and the viewport."""

        def render(date, layer, cmap, opacity, clim, x_range=None, y_range=None, width=None, height=None,
                   scale=1.):
            element = self._render(date, layer, title_func(date, layer), x_range, y_range, width, height)
            if isinstance(element, hv.RGB):
                return element.opts(alpha=opacity)
            return element.opts(cmap=cc.cm[cmap], alpha=opacity, clim=tuple(clim))

        image = hv.DynamicMap(
            pn.bind(render, date=date_widget, layer=layer_widget, cmap=cmap_widget,
                    opacity=opacity_widget, clim=range_widget),
            streams=[hv.streams.RangeXY(), hv.streams.PlotSize()]
        ).opts(opts.Image(**self._image_opts()), opts.RGB(width=self.width, height=self.height,
                                                          active_tools=['wheel_zoom']))
        if not self.reproject:
            return image * self.aoi_polygons
        tiles = hv.DynamicMap(pn.bind(self._tiles, basemap=basemap_widget))
        return tiles * image * self.aoi_polygons

    def _render(self, date, layer, title, x_range, y_range, width, height):
        return self._image(date, layer, title, x_range, y_range, width, height)


class ViewSpectral(_MultiscaleViewer):
    """
    Interactive viewer of L2A images (Rrs with 'wl' and optional 'time' dimensions): true-colour
    composite or single bands, areas of interest (polygons) and points of interest whose
    full-resolution spectra are plotted.

    Only the pixels visible on the screen are read, from the coarsest resolution level that matches
    the zoom: the pyramids of the GRS Zarr products are used directly, and coarser levels are computed
    on demand for the other inputs (see :class:`Multiscale`).

    Example
    -------
    >>> viewer = ViewSpectral(['S2A_L2A_1.zarr', 'S2B_L2A_2.zarr'], reproject=True)   # Zarr pyramids
    >>> viewer = ViewSpectral(datacube.Rrs, reproject=True)                         # datacube
    >>> viewer = ViewSpectral(datacube.Rrs, reproject=True, basemap='CARTO Positron')
    >>> viewer.visu()
    >>> aoi = viewer.get_geom()            # last drawn polygon, EPSG:4326
    >>> points = viewer.get_points()       # drawn points, EPSG:4326

    :param raster: Rrs DataArray with dims (time, wl, y, x) (time is optional), L2A Dataset, path or
                   list of paths of Zarr stores, or :class:`Multiscale`
    :param dates: dates to display (default: all)
    :param bands: wavelengths to display (default: all)
    :param reproject: display in web Mercator (EPSG:3857) to overlay basemaps; only the visible
                      window is reprojected
    :param basemap: initial basemap, name in :data:`BASEMAPS` (e.g. 'CARTO Positron') or XYZ URL
                    with {X}, {Y}, {Z}; it can be changed in the widget
    :param minmaxvalues: initial color range
    :param minmax: bounds of the color range slider
    :param rgb_bands: wavelengths of the true-colour composite
    :param gamma: exponent applied to Rrs in the composite to enhance dark (water) pixels
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    """

    def __init__(self, raster, dates=None,
                 bands=None,
                 reproject=False,
                 minmaxvalues=(0, 0.02),
                 minmax=(0, 0.06),
                 rgb_bands=(665, 560, 490),
                 gamma=0.5,
                 width=1000,
                 height=700,
                 basemap=DEFAULT_BASEMAP):

        super().__init__(raster, dates=dates, reproject=reproject, minmaxvalues=minmaxvalues, minmax=minmax,
                         width=width, height=height, basemap=basemap)
        self.title = '## S2 L2A'

        image = self.data.images[self.indexes[0]][0]
        wls = (image['Rrs'] if isinstance(image, xr.Dataset) else image).wl.values
        self.wls = wls
        self.bands = wls if bands is None else np.asarray(bands)
        self.rgb_bands = [wls[np.abs(wls - wl).argmin()] for wl in rgb_bands]
        self.gamma = gamma
        self._stretch = {}

        # declare streaming object to get Point of Interest (POI), in the CRS of the map
        self.poi_points = hv.Points([], vdims='color').opts(opts.Points(
            active_tools=['point_draw'], color='color', size=8, line_color='white'))
        self.poi_stream = hv.streams.PointDraw(data=self.poi_points.columns(), source=self.poi_points,
                                               empty_value='red')
        self.table = hv.Table(self.poi_points, ['x', 'y'], 'color').opts(opts.Table(editable=True, height=200))
        DataLink(self.poi_points, self.table)

    def _rgb_stretch(self, idate):
        """Percentiles (2, 98) of the composite, from the coarsest level (one per image)."""
        if idate not in self._stretch:
            image = self.indexes[idate]
            rgb = self.data.layer(image, self.rgb_bands, self.data.nlevels(image) - 1).load()
            rgb = rgb.clip(min=0) ** self.gamma
            self._stretch[idate] = (float(rgb.quantile(0.02)), float(rgb.quantile(0.98)))
        return self._stretch[idate]

    def _render(self, date, layer, title, x_range, y_range, width, height):
        if layer != 'RGB':
            return self._image(date, layer, title, x_range, y_range, width, height)

        x_range, y_range = self._ranges(date, x_range, y_range)
        da = self._read(date, self.rgb_bands, x_range, y_range, width or self.width, height or self.height)
        if da is None:
            return hv.RGB(np.zeros((1, 1, 4)), bounds=(x_range[0], y_range[0], x_range[1], y_range[1]),
                          vdims=['R', 'G', 'B', 'A']).opts(title=title)
        vmin, vmax = self._rgb_stretch(date)
        da = da.transpose('wl', 'y', 'x')
        if da.y[0] < da.y[-1]:
            da = da.isel(y=slice(None, None, -1))
        values = ((da.values.clip(min=0) ** self.gamma - vmin) / (vmax - vmin)).clip(0, 1)
        alpha = np.isfinite(values).all(axis=0).astype(float)
        rgba = np.dstack([np.nan_to_num(values[i]) for i in range(3)] + [alpha])
        x, y = da.x.values, da.y.values
        dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
        return hv.RGB(rgba, bounds=(x.min() - dx, y.min() - dy, x.max() + dx, y.max() + dy),
                      vdims=['R', 'G', 'B', 'A']).opts(title=title)

    def _spectra(self, data, date):
        """Full-resolution spectra of the points drawn on the map."""
        label = r'Rrs (sr-1)'
        curves = {}
        if data:
            for ipoint, (x, y, color) in enumerate(zip(data.get('x', []), data.get('y', []),
                                                       data.get('color', []))):
                spectrum = self.data.pixel(self.indexes[date], *self._to_native.transform(x, y))
                if spectrum is None:
                    continue
                curves[ipoint] = hv.Curve((spectrum.wl.values, spectrum.values), 'Wavelength (nm)',
                                          label).opts(color=color or 'red')
        if not curves:
            curves[0] = hv.Curve([], 'Wavelength (nm)', label)
        return hv.NdOverlay(curves, kdims='point').opts(
            opts.Curve(width=450, height=350, tools=['hover'], line_width=2, show_grid=True),
            opts.NdOverlay(show_legend=False, title='Spectra of the points (full resolution)'))

    def visu(self):
        """Return the panel layout of the viewer."""

        layers = {'RGB': 'RGB'}
        layers.update({'{:.0f}'.format(wl): wl for wl in self.bands})
        pn_band = pn.widgets.RadioButtonGroup(value='RGB', options=layers)
        pn_date = self._date_widget()
        pn_colormap = pn.widgets.Select(value='CET_D13', options=self.colormaps)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0],
                                                      end=self.minmax[1], value=self.minmaxvalues, step=0.0001)
        pn_basemaps = self._basemap_widget()

        def title(date, layer):
            band = 'RGB' if layer == 'RGB' else 'wl = {:.0f} nm'.format(layer)
            return '{}, {}'.format(self.datetimes[date], band)

        map_ = self._map(pn_band, pn_date, pn_colormap, pn_opacity, range_slider, pn_basemaps, title)
        spectra = hv.DynamicMap(pn.bind(self._spectra, date=pn_date), streams=[self.poi_stream])

        return pn.Column(
            pn.WidgetBox(
                self.title,
                pn.Column(
                    pn.Row('### Band', pn_band),
                    pn.Row(
                        pn.Row('### Date', pn_date),
                        pn.Row('#### Basemap', pn_basemaps)
                    ),
                    pn.Row(
                        pn.Row('', range_slider),
                        pn.Row('#### Opacity', pn_opacity),
                        pn.Row('#### Colormap', pn_colormap))
                ),
            ),
            pn.Row(
                pn.pane.HoloViews((map_ * self.poi_points).opts(
                    opts.Points(active_tools=['point_draw'], color='color'))),
                pn.Column(pn.pane.HoloViews(spectra), pn.pane.HoloViews(self.table))
            )
        )


class ViewParam(_MultiscaleViewer):
    """
    Interactive viewer of L2B images (one 2D map per parameter and date).

    Only the pixels visible on the screen are read, from the coarsest resolution level that matches
    the zoom (see :class:`ViewSpectral` and :class:`Multiscale`).

    Example
    -------
    >>> viewer = ViewParam(['S2A_L2B_1.zarr', 'S2B_L2B_2.zarr'], reproject=True)
    >>> viewer = ViewParam(datacube, params=['SPM_nechad', 'Chla_OC2nasa'], reproject=True)
    >>> viewer.visu()

    :param raster: Dataset with dims (time, y, x) (time is optional), path or list of paths of Zarr
                   stores, or :class:`Multiscale`
    :param dates: dates to display (default: all)
    :param params: variables to display (default: all variables with x and y dimensions)
    :param reproject: display in web Mercator (EPSG:3857) to overlay basemaps; only the visible
                      window is reprojected
    :param basemap: initial basemap, name in :data:`BASEMAPS` (e.g. 'CARTO Positron') or XYZ URL
                    with {X}, {Y}, {Z}; it can be changed in the widget
    :param minmaxvalues: initial color range
    :param minmax: bounds of the color range slider
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    """

    def __init__(self, raster, dates=None,
                 params=None,
                 reproject=False,
                 minmaxvalues=(0, 4),
                 minmax=(0, 10),
                 width=1200,
                 height=800,
                 basemap=DEFAULT_BASEMAP):

        super().__init__(raster, dates=dates, reproject=reproject, minmaxvalues=minmaxvalues, minmax=minmax,
                         width=width, height=height, basemap=basemap)
        self.title = '## S2 L2B'

        image = self.data.images[self.indexes[0]][0]
        self.params = params
        if params is None:
            # keep the 2D maps
            self.params = [param for param in image.data_vars
                           if set(image[param].dims) == {'x', 'y'}]

    def visu(self):
        """Return the panel layout of the viewer."""

        pn_param = pn.widgets.Select(value=self.params[0], options=self.params)
        pn_date = self._date_widget()
        pn_colormap = pn.widgets.Select(value='CET_D13', options=self.colormaps)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0],
                                                      end=self.minmax[1], value=self.minmaxvalues, step=0.0001)
        pn_basemaps = self._basemap_widget()

        def title(date, param):
            return '{}, {}'.format(self.datetimes[date], param)

        map_ = self._map(pn_param, pn_date, pn_colormap, pn_opacity, range_slider, pn_basemaps, title)

        return pn.Column(
            pn.WidgetBox(
                self.title,
                pn.Column(
                    pn.Row(
                        pn.Row('### Parameter', pn_param),
                        pn.Row('### Date', pn_date),
                        pn.Row('#### Basemap', pn_basemaps)
                    ),
                    pn.Row(range_slider,
                           pn.Row('#### Opacity', pn_opacity),
                           pn.Row('#### Colormap', pn_colormap)
                           )
                ),
            ),
            pn.pane.HoloViews(map_)
        )
