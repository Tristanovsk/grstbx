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
import matplotlib
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401, activates the .rio accessor
import pyproj
import geopandas as gpd

import holoviews as hv
from holoviews.element import tiles as hvts
from holoviews import opts

import bokeh
import colorcet as cc
import panel as pn
import param as pm
import shapely.wkt
from shapely.geometry import Point, Polygon

hv.extension('bokeh')

# Free basemaps (XYZ raster tiles, web Mercator) proposed in the ViewSpectral / ViewParam widgets.
# The Stamen styles are now served by Stadia Maps: free without key when the notebook runs on
# localhost, otherwise a (free) API key is needed, see set_stadia_api_key. The default basemap
# needs no key, so that maps also display on a server or in a published page (e.g. Read the Docs).
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
DEFAULT_BASEMAP = 'OpenStreetMap'
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


# Colormaps of the ViewSpectral and ViewParam widgets: sequential and perceptually uniform (Rrs is a positive
# quantity with no meaningful centre, diverging or cyclic maps are misleading), named after the
# water feature they enhance. Values: colorcet name, 'cmo.<name>' (cmocean) or matplotlib name.
RRS_COLORMAPS = {
    'Rainbow (CET_R1)': 'CET_R1',
    'Turbid water, sediments (cmocean turbid)': 'cmo.turbid',
    'Clear / shallow water (cmocean deep)': 'cmo.deep_r',
    'Algae, chlorophyll (cmocean algae)': 'cmo.algae',
    'Water masses (cmocean haline)': 'cmo.haline',
    'Gouldian (CET_L20)': 'CET_L20',
    'Viridis': 'viridis',
    'Cividis (colour-blind safe)': 'cividis',
    'Blues (CET_L12)': 'blues',
    'Fire (CET_L3)': 'fire',
    'Grey (CET_L1)': 'gray',
}
DEFAULT_RRS_COLORMAP = 'Rainbow (CET_R1)'

# Colours of the points drawn on the viewers, cycled in the drawing order (Category10 without grey,
# kept for the black live curve and the grey date line)
POINT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22',
                '#17becf']

# Band combinations proposed for the RGB composite of ViewSpectral (wavelengths in nm of the red, green and
# blue channels), kept when the image has bands within RGB_PRESET_TOLERANCE nm of them
RGB_PRESETS = {
    'True colour': (665, 560, 490),
    'False colour (NIR, vegetation)': (842, 665, 560),
    'Red edge (phytoplankton)': (705, 560, 490),
    'SWIR (land / water, clouds)': (2190, 865, 665),
}
RGB_PRESET_TOLERANCE = 30

# Style of the widget boxes of the viewers: the drop-down menus (colormap picker, selects) may overflow the
# box, above the map below it, instead of being clipped by the box
_FRONT_CSS = ':host { overflow: visible !important; position: relative; z-index: 10; }'


def _to_front(box):
    """Let the drop-down menus of the widgets of the layout ``box`` overflow it, above the plots below."""
    layouts = {id(box): box}
    layouts.update({id(layout): layout for layout in box.select(pn.layout.ListPanel)})
    for layout in layouts.values():
        layout.stylesheets = list(layout.stylesheets) + [_FRONT_CSS]
    return box


def get_colormap(cmap, n=256):
    """
    List of hex colours of a colormap.

    :param cmap: list of colours, label of :data:`RRS_COLORMAPS`, colorcet name, 'cmo.<name>'
                 (cmocean) or matplotlib name; '_r' suffix to reverse
    :param n: number of colours
    :return: list of hex colours
    """
    if isinstance(cmap, (list, tuple)):
        return list(cmap)
    cmap = RRS_COLORMAPS.get(cmap, cmap)
    if cmap in cc.cm:
        cmap = cc.cm[cmap]
    elif cmap.startswith('cmo.'):
        import cmocean  # noqa: F401, registers the 'cmo.*' colormaps in matplotlib
        cmap = matplotlib.colormaps[cmap]
    else:
        cmap = matplotlib.colormaps[cmap]
    return [matplotlib.colors.to_hex(c) for c in cmap(np.linspace(0, 1, n))]


def color_scale_hook(log):
    """
    Holoviews hook switching the color scale of an image between linear and log.

    Holoviews creates the bokeh color mapper at the first rendering and only updates its range
    afterwards: changing ``logz`` in a DynamicMap has no effect. When the type of the mapper does
    not match ``log``, the hook replaces the glyph renderer by a copy using a new mapper. Changing
    the mapper of the existing glyph is not enough: bokehjs draws copies of the glyph made at the
    first rendering (decimated, selection, muted glyphs) which keep the initial mapper.

    :param log: True for a log color scale, False for a linear one
    """
    from bokeh.models import BasicTicker, LinearColorMapper, LogColorMapper, LogTicker
    mapper_type, ticker_type = (LogColorMapper, LogTicker) if log else (LinearColorMapper, BasicTicker)

    def copy(model, **changes):
        return type(model)(**{**model.properties_with_values(include_defaults=False), **changes})

    def hook(plot, element):
        mapper = plot.handles.get('color_mapper')
        renderer = plot.handles.get('glyph_renderer')
        if mapper is None or renderer is None or type(mapper) is mapper_type:
            return
        new = mapper_type(palette=mapper.palette, low=mapper.low, high=mapper.high, nan_color=mapper.nan_color,
                          low_color=mapper.low_color, high_color=mapper.high_color)
        glyphs = {name: copy(getattr(renderer, name), color_mapper=new)
                  for name in ('glyph', 'selection_glyph', 'nonselection_glyph', 'hover_glyph', 'muted_glyph')
                  if getattr(getattr(renderer, name), 'color_mapper', None) is mapper}
        new_renderer = copy(renderer, **glyphs)

        fig = plot.state
        fig.renderers = [new_renderer if r is renderer else r for r in fig.renderers]
        for tool in fig.tools:
            if isinstance(getattr(tool, 'renderers', None), list) and renderer in tool.renderers:
                tool.renderers = [new_renderer if r is renderer else r for r in tool.renderers]
        plot.handles.update(color_mapper=new, glyph=glyphs['glyph'], glyph_renderer=new_renderer)
        if 'colorbar' in plot.handles:
            # ticker first: bokehjs redraws the colorbar when its mapper changes, not its ticker
            colorbar = plot.handles['colorbar']
            colorbar.ticker = ticker_type()
            colorbar.color_mapper = new

    return hook


def colorbar_visibility_hook(visible):
    """
    Holoviews hook showing or hiding the colorbar of an image. The colorbar is created at the first
    rendering and ``colorbar=False`` in a DynamicMap does not remove it: the viewers hide it when the RGB
    composite is displayed (no colour scale) and show it again for the single bands and parameters.

    :param visible: True to show the colorbar, False to hide it
    """
    def hook(plot, element):
        colorbar = plot.handles.get('colorbar')
        if colorbar is not None and colorbar.visible != visible:
            colorbar.visible = visible

    return hook


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


class _LRUCache(OrderedDict):
    """
    Least-recently-used cache of arrays, bounded in memory: the oldest items are dropped when the total
    size of the cached arrays exceeds ``max_bytes`` (one item is always kept).
    """

    def __init__(self, max_bytes):
        super().__init__()
        self.max_bytes = max_bytes
        self.nbytes = 0

    @staticmethod
    def _size(value):
        if isinstance(value, tuple):
            return sum(_LRUCache._size(item) for item in value)
        return int(getattr(value, 'nbytes', 0) or 0)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key, value):
        if key in self:
            self.nbytes -= self._size(super().pop(key))
        self[key] = value
        self.nbytes += self._size(value)
        while self.nbytes > self.max_bytes and len(self) > 1:
            self.nbytes -= self._size(self.popitem(last=False)[1])
        return value


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
    :param cache_size: memory (bytes) of the data kept in cache: windows read for the map (with a margin,
                       so that panning and zooming read the data again only when leaving it), coarser
                       levels computed for the images without pyramid, and blocks read for the spectra
                       and time series; the least recently used data are dropped first
    """

    #: margin added on each side of the windows read for the map (fraction of their width and height)
    window_margin = 0.25

    def __init__(self, images, times, min_size=512, cache_size=2e9):
        self.images = images
        self.times = np.asarray(times)
        self.min_size = min_size
        self.crs = images[0][0].rio.crs
        self._bounds = {}
        # coarser levels computed on demand, windows read for the map, and full-resolution blocks (all
        # wavelengths) read for the spectra and time series, kept in memory within cache_size
        self._cache = _LRUCache(cache_size * 0.25)
        self._windows = _LRUCache(cache_size * 0.25)
        self._blocks = _LRUCache(cache_size * 0.5)

    @classmethod
    def from_xarray(cls, obj, min_size=512, cache_size=2e9):
        """
        :param obj: DataArray (e.g. ``Rrs`` with a ``wl`` dimension) or Dataset with x, y and optional
                    ``time`` dimensions
        """
        if 'time' not in obj.dims:
            obj = obj.expand_dims('time')
        images = [[obj.isel(time=itime)] for itime in range(obj.sizes['time'])]
        return cls(images, obj.time.values, min_size=min_size, cache_size=cache_size)

    @classmethod
    def from_zarr(cls, paths, min_size=512, cache_size=2e9):
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
        return cls([images[i] for i in order], np.array(times)[order], min_size=min_size, cache_size=cache_size)

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
        if idate not in self._bounds:
            self._bounds[idate] = self.images[idate][0].rio.bounds()
        return self._bounds[idate]

    def layer(self, idate, name, level=0):
        """Lazy (stored levels) or in-memory (computed levels) 2-D or 3-D layer of an image."""
        levels = self.images[idate]
        if level < len(levels):
            return self._select(levels[level], name)
        key = (idate, tuple(name) if isinstance(name, (list, tuple)) else name, level)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cache.put(key, _coarsen2(self.layer(idate, name, level - 1)).load())
        return cached

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
        res = self.resolution(idate, level)
        key = (idate, tuple(name) if isinstance(name, (list, tuple)) else name, level)

        def crop(da, xmin, ymin, xmax, ymax):
            yslice = slice(ymax + res, ymin - res) if da.y[0] > da.y[-1] else slice(ymin - res, ymax + res)
            return da.sel(x=slice(xmin - res, xmax + res), y=yslice)

        # the last window read at this level, with its margin: panning and zooming inside it read nothing
        # (only the part of the requested area on the image matters: there is no data outside)
        exmin, eymin, exmax, eymax = self.bounds(idate)
        inside = (max(bounds[0], exmin), max(bounds[1], eymin), min(bounds[2], exmax), min(bounds[3], eymax))
        cached = self._windows.get(key)
        if cached is not None:
            (cxmin, cymin, cxmax, cymax), da = cached
            if cxmin <= inside[0] and cymin <= inside[1] and inside[2] <= cxmax and inside[3] <= cymax:
                return crop(da, *bounds), level

        xmin, ymin, xmax, ymax = bounds
        mx, my = (xmax - xmin) * self.window_margin, (ymax - ymin) * self.window_margin
        expanded = (max(xmin - mx, exmin), max(ymin - my, eymin), min(xmax + mx, exmax), min(ymax + my, eymax))
        # the window keeps the requested area even when it overlaps the edge of the image
        expanded = (min(expanded[0], xmin), min(expanded[1], ymin), max(expanded[2], xmax), max(expanded[3], ymax))
        da = crop(self.layer(idate, name, level), *expanded).load()
        self._windows.put(key, (expanded, da))
        return crop(da, *bounds), level

    @staticmethod
    def _block_size(da, dim, default=256, max_size=512):
        """Size of the blocks read for the spectra: chunk size of the store, bounded."""
        size = None
        if da.chunks:
            size = da.chunks[da.get_axis_num(dim)][0]
        elif dim in da.encoding.get('preferred_chunks', {}):
            size = da.encoding['preferred_chunks'][dim]
        return int(min(size or default, max_size))

    def pixel(self, idate, x, y, name=None):
        """
        Full-resolution values (all wavelengths for Rrs) of the pixel nearest to (x, y), or None outside.

        The block of pixels around (x, y), aligned on the chunks of the store, is read once and kept in
        memory, so that successive calls nearby (e.g. when hovering the map) do not read the data again.

        :param name: layer to read (variable name or wavelength, see :meth:`layer`); default: Rrs
                     (all wavelengths) or the whole image
        """
        obj = self.images[idate][0]
        if name is not None:
            da = self._select(obj, name)
        else:
            da = obj['Rrs'] if isinstance(obj, xr.Dataset) and 'Rrs' in obj else obj
        xmin, ymin, xmax, ymax = self.bounds(idate)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return None
        xs, ys = da.x.values, da.y.values
        ix = int(np.clip(np.rint((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
        iy = int(np.clip(np.rint((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
        bx, by = self._block_size(da, 'x'), self._block_size(da, 'y')
        key = (idate, name, ix // bx, iy // by)
        block = self._blocks.get(key)
        if block is None:
            block = self._blocks.put(key, da.isel(x=slice(key[2] * bx, (key[2] + 1) * bx),
                                                  y=slice(key[3] * by, (key[3] + 1) * by)).load())
        return block.isel(x=ix % bx, y=iy % by)

    def timeseries(self, name, x, y, indexes=None):
        """
        Full-resolution time series of a layer at the pixel nearest to (x, y).

        :param name: variable name or wavelength
        :param indexes: indexes of the images (default: all)
        :return: pandas.Series indexed by acquisition time (images not covering (x, y) are skipped)
        """
        indexes = range(len(self.images)) if indexes is None else indexes
        times, values = [], []
        for idate in indexes:
            value = self.pixel(idate, x, y, name)
            if value is not None:
                times.append(self.times[idate])
                values.append(float(value))
        return pd.Series(values, index=pd.DatetimeIndex(times), name=name, dtype=float)


class _MultiscaleViewer(Utils):
    """
    Base of the viewers: map rendered from the pixels visible on the screen only.

    At each pan / zoom, the visible window is read at the coarsest resolution level that still has at
    least one image pixel per screen pixel (see :class:`Multiscale`) and, with ``reproject=True``,
    reprojected to web Mercator (EPSG:3857) to overlay basemaps. The last windows are kept in memory
    so that changing the colormap, the color range or the opacity does not read the data again.
    """

    def __init__(self, data, dates=None, reproject=False, minmaxvalues=(0, 0.02), minmax=(0, 0.06),
                 width=1200, height=800, min_size=512, basemap=DEFAULT_BASEMAP, cmap=DEFAULT_RRS_COLORMAP,
                 aoi=None, poi=None):
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

        # colormaps of the picker (label -> colours); a custom colormap is added to the list
        self.colormaps = {label: get_colormap(name) for label, name in RRS_COLORMAPS.items()}
        labels = {name: label for label, name in RRS_COLORMAPS.items()}
        cmap = labels.get(cmap, cmap)   # e.g. 'cmo.turbid' -> its label
        if cmap not in self.colormaps:
            self.colormaps[str(cmap)] = get_colormap(cmap)
        self.cmap = str(cmap)

        # display CRS and transformations from / to the CRS of the images
        self.reproject = reproject
        self.basemap = basemap
        self.display_crs = pyproj.CRS.from_epsg(3857) if reproject else pyproj.CRS.from_user_input(data.crs)
        self._to_native = pyproj.Transformer.from_crs(self.display_crs, data.crs, always_xy=True)
        self._to_display = pyproj.Transformer.from_crs(data.crs, self.display_crs, always_xy=True)
        self._windows = OrderedDict()
        self._max_windows = 16

        # contrast and brightness of the map (see :meth:`display_controls`)
        self.pn_contrast = pn.widgets.FloatSlider(name='Contrast', value=1., start=0.2, end=4., step=0.05)
        self.pn_brightness = pn.widgets.FloatSlider(name='Brightness', value=0., start=-0.5, end=0.5, step=0.01)

        # declare streaming object to get Area of Interest (AOI), in the CRS of the map, with the
        # initial polygons (if any)
        xs, ys = self._aoi_coords(aoi)
        self.aoi_polygons = hv.Polygons([{'x': x, 'y': y} for x, y in zip(xs, ys)]).opts(opts.Polygons(
            fill_alpha=0.3, fill_color='white', line_width=1.2))
        self.aoi_stream = hv.streams.PolyDraw(source=self.aoi_polygons, drag=True,
                                              data={'xs': xs, 'ys': ys} if xs else None)
        self.edit_stream = hv.streams.PolyEdit(source=self.aoi_polygons, vertex_style={'color': 'red'})

        # declare streaming object to get Points of Interest (POI), in the CRS of the map
        # (colours cycled in the drawing order, see :meth:`point_color`), with the initial points (if any)
        points = self._display_geoms(poi)
        self.poi_points = hv.Points(
            {'x': list(points.x), 'y': list(points.y), 'color': [self.point_color(i) for i in range(len(points))]},
            vdims='color').opts(opts.Points(
                active_tools=['point_draw'], color='color', size=8, line_color='white',
                hooks=[self._cycle_point_colors]))
        self.poi_stream = hv.streams.PointDraw(data=self.poi_points.columns(), source=self.poi_points,
                                               empty_value=POINT_COLORS[0])

    def display_controls(self):
        """
        Contrast and brightness sliders of the map, with a reset button, e.g. to place them in another
        layout (see :mod:`grstbx.app`). On the RGB composite, they act on the stretched values (0-1); on the
        single bands or parameters, they narrow (contrast > 1) and shift (brightness) the colour range.
        """
        reset = pn.widgets.Button(name='Reset', button_type='light')

        def on_reset(event):
            self.pn_contrast.value, self.pn_brightness.value = 1., 0.

        reset.on_click(on_reset)
        return pn.Column(self.pn_contrast, self.pn_brightness, reset, sizing_mode='stretch_width')

    @staticmethod
    def _adjust_clim(clim, contrast=1., brightness=0.):
        """Colour range narrowed by ``contrast`` around its centre, shifted down by ``brightness`` (fraction
        of the range) so that the image looks brighter."""
        vmin, vmax = clim
        span = vmax - vmin
        centre = (vmin + vmax) / 2 - brightness * span
        return centre - span / 2 / contrast, centre + span / 2 / contrast

    def _rgb_bindings(self):
        """Widgets (or bound functions) of the RGB composite passed to :meth:`_render` (none here)."""
        return {}

    # -- geometries drawn on the map (in the CRS of the map)

    def _display_geoms(self, geoms):
        """
        Initial geometries (``aoi`` or ``poi`` parameters) as a GeoSeries in the CRS of the map.

        :param geoms: GeoDataFrame or GeoSeries (with its CRS), shapely geometry, WKT string, (lon, lat)
                      tuple (points) or list of them (EPSG:4326)
        """
        if geoms is None:
            return gpd.GeoSeries([], crs=self.display_crs)
        if not isinstance(geoms, (gpd.GeoDataFrame, gpd.GeoSeries)):
            def is_lonlat(g):
                return isinstance(g, (list, tuple)) and len(g) == 2 and all(isinstance(v, (int, float, np.number)) for v in g)

            if not isinstance(geoms, (list, tuple)) or is_lonlat(geoms):
                geoms = [geoms]
            geoms = gpd.GeoSeries([shapely.wkt.loads(g) if isinstance(g, str) else
                                   Point(g) if is_lonlat(g) else g for g in geoms], crs=4326)
        return geoms.geometry.to_crs(self.display_crs).explode(index_parts=False)

    def _aoi_coords(self, aoi):
        """Exterior coordinates (xs, ys lists, in the CRS of the map) of the initial polygons."""
        polygons = self._display_geoms(aoi)
        xs = [list(polygon.exterior.xy[0]) for polygon in polygons]
        ys = [list(polygon.exterior.xy[1]) for polygon in polygons]
        return xs, ys

    def get_geom(self, aoi_stream=None, crs=4326, index=-1):
        """Polygon drawn on the map (default: last one) as a GeoDataFrame in ``crs``."""
        return Utils.get_geom(aoi_stream or self.aoi_stream, crs=crs, index=index, src_crs=self.display_crs)

    def get_points(self, poi_stream=None, crs=4326):
        """Points drawn on the map as a GeoDataFrame in ``crs``."""
        return Utils.get_points(poi_stream or self.poi_stream, crs=crs, src_crs=self.display_crs)

    @staticmethod
    def point_color(ipoint):
        """Colour of the point ``ipoint`` (drawing order) and of its curve."""
        return POINT_COLORS[ipoint % len(POINT_COLORS)]

    @staticmethod
    def _cycle_point_colors(plot, element):
        """Bokeh hook: colour the points in the browser as they are drawn (see :meth:`point_color`)."""
        source = plot.handles['source']
        if source.tags and '_cycle_point_colors' in source.tags:
            return
        source.tags = list(source.tags) + ['_cycle_point_colors']
        # the colours are written back to the data, which are synced to the PointDraw stream
        source.js_on_change('data', bokeh.models.CustomJS(args=dict(source=source, palette=POINT_COLORS), code='''
            const colors = source.data.color
            if (colors == null)
                return
            const cycled = Array.from(colors, (_, i) => palette[i % palette.length])
            if (cycled.some((color, i) => color !== colors[i]))
                source.data = Object.assign({}, source.data, {color: cycled})
        '''))

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

    def _colormap_widgets(self):
        """Colormap picker with swatches, reverse toggle and the bound list of colours."""
        pn_colormap = pn.widgets.ColorMap(options=self.colormaps, value_name=self.cmap, swatch_width=120,
                                          width=260)
        pn_reverse = pn.widgets.Checkbox(name='Reverse', value=False)
        cmap = pn.bind(lambda colors, reverse: list(colors)[::-1] if reverse else list(colors),
                       pn_colormap, pn_reverse)
        return pn_colormap, pn_reverse, cmap

    def _map(self, layer_widget, date_widget, cmap_widget, opacity_widget, range_widget, basemap_widget,
             title_func, log_widget=False):
        """Overlay of the basemap and of the multiscale image driven by the widgets and the viewport."""

        # a DynamicMap must always return the same element type: single layers (hv.Image) and
        # composites (hv.RGB) are drawn by two DynamicMaps, the inactive one returns an empty element
        def render_image(date, layer, cmap, opacity, clim, log, contrast=1., brightness=0., x_range=None,
                         y_range=None, width=None, height=None, scale=1.):
            if self._is_rgb(layer):
                x_range, y_range = self._ranges(date, x_range, y_range)
                return hv.Image(np.zeros((1, 1)), vdims=['image'],
                                bounds=(x_range[0], y_range[0], x_range[1], y_range[1])).opts(
                    alpha=0, hooks=[colorbar_visibility_hook(False)])
            element = self._render(date, layer, title_func(date, layer), x_range, y_range, width, height)
            vmin, vmax = self._adjust_clim(clim, contrast, brightness)
            if log and vmin <= 0:
                # a log color scale needs a positive lower bound
                vmin = vmax * 1e-3 if vmax > 0 else 1e-6
            return element.opts(cmap=get_colormap(cmap), alpha=opacity, clim=(vmin, vmax), logz=log,
                                colorbar=True, hooks=[color_scale_hook(log), colorbar_visibility_hook(True)])

        def render_rgb(date, layer, opacity, contrast=1., brightness=0., x_range=None, y_range=None, width=None,
                       height=None, scale=1., **style):
            if not self._is_rgb(layer):
                x_range, y_range = self._ranges(date, x_range, y_range)
                return hv.RGB(np.zeros((1, 1, 4)), vdims=['R', 'G', 'B', 'A'],
                              bounds=(x_range[0], y_range[0], x_range[1], y_range[1]))
            element = self._render(date, layer, title_func(date, layer), x_range, y_range, width, height,
                                   contrast=contrast, brightness=brightness, **style)
            return element.opts(alpha=opacity)

        image = hv.DynamicMap(
            pn.bind(render_image, date=date_widget, layer=layer_widget, cmap=cmap_widget,
                    opacity=opacity_widget, clim=range_widget, log=log_widget, contrast=self.pn_contrast,
                    brightness=self.pn_brightness),
            streams=[hv.streams.RangeXY(), hv.streams.PlotSize()]
        ).opts(opts.Image(**self._image_opts()))
        if self.has_rgb:
            rgb = hv.DynamicMap(
                pn.bind(render_rgb, date=date_widget, layer=layer_widget, opacity=opacity_widget,
                        contrast=self.pn_contrast, brightness=self.pn_brightness, **self._rgb_bindings()),
                streams=[hv.streams.RangeXY(), hv.streams.PlotSize()]
            ).opts(opts.RGB(width=self.width, height=self.height, active_tools=['wheel_zoom']))
            image = rgb * image
        if not self.reproject:
            return image * self.aoi_polygons
        tiles = hv.DynamicMap(pn.bind(self._tiles, basemap=basemap_widget))
        return tiles * image * self.aoi_polygons

    has_rgb = False

    def _is_rgb(self, layer):
        return False

    def _render(self, date, layer, title, x_range, y_range, width, height, **style):
        return self._image(date, layer, title, x_range, y_range, width, height)


class ViewSpectral(_MultiscaleViewer):
    """
    Interactive viewer of L2A images (Rrs with 'wl' and optional 'time' dimensions): true-colour
    composite or single bands, areas of interest (polygons) and points of interest whose
    full-resolution spectra are plotted. The spectrum of the pixel under the mouse is also plotted
    live (dashed line), which can be turned off with the 'Live spectrum' checkbox. The spectra are
    displayed beneath the map.

    Only the pixels visible on the screen are read, from the coarsest resolution level that matches
    the zoom: the pyramids of the GRS Zarr products are used directly, and coarser levels are computed
    on demand for the other inputs (see :class:`Multiscale`).

    Example
    -------
    >>> viewer = ViewSpectral(['S2A_L2A_1.zarr', 'S2B_L2A_2.zarr'], reproject=True)   # Zarr pyramids
    >>> viewer = ViewSpectral(datacube.Rrs, reproject=True)                         # datacube
    >>> viewer = ViewSpectral(datacube.Rrs, reproject=True, basemap='CARTO Positron', cmap='cmo.turbid')
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
    :param cmap: initial colormap of the single bands, label of :data:`RRS_COLORMAPS`
                 (e.g. 'Turbid water, sediments (cmocean turbid)'), or any colorcet, cmocean
                 ('cmo.<name>') or matplotlib name
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :param aoi: initial polygons of the areas of interest, returned by :meth:`get_geom` until other
                polygons are drawn (e.g. for non-interactive runs): GeoDataFrame or GeoSeries, shapely
                (Multi)Polygon, WKT string or list of them (EPSG:4326)
    :param poi: initial points of interest, returned by :meth:`get_points` (and plotted) until they are
                moved or deleted: GeoDataFrame or GeoSeries, shapely Point, WKT string, (lon, lat) tuple
                or list of them (EPSG:4326)
    :param title: title of the viewer, in Markdown (e.g. '## PRISMA L2A', '## Landsat 9 L2A'),
                  default '## S2 L2A'
    :param live: initial state of the *Live spectrum* checkbox
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
                 basemap=DEFAULT_BASEMAP,
                 cmap=DEFAULT_RRS_COLORMAP,
                 aoi=None,
                 poi=None,
                 title='## S2 L2A',
                 live=True):

        super().__init__(raster, dates=dates, reproject=reproject, minmaxvalues=minmaxvalues, minmax=minmax,
                         width=width, height=height, basemap=basemap, cmap=cmap, aoi=aoi,
                         poi=poi)
        self.title = title
        self.live = live

        image = self.data.images[self.indexes[0]][0]
        wls = (image['Rrs'] if isinstance(image, xr.Dataset) else image).wl.values
        self.wls = wls
        self.bands = wls if bands is None else np.asarray(bands)
        self.rgb_bands = [self._nearest_band(wl) for wl in rgb_bands]
        self.gamma = gamma
        self._stretch = {}

        # bands and gamma of the RGB composite (see :meth:`rgb_controls`)
        options = {self._band_label(wl): wl for wl in wls}
        self.pn_red, self.pn_green, self.pn_blue = [
            pn.widgets.Select(name=name, options=options, value=wl, width=100)
            for name, wl in zip(['Red', 'Green', 'Blue'], self.rgb_bands)]
        self.rgb_presets = {name: tuple(self._nearest_band(wl) for wl in bands) for name, bands in RGB_PRESETS.items()
                            if all(abs(self._nearest_band(wl) - wl) <= RGB_PRESET_TOLERANCE for wl in bands)}
        current = next((name for name, bands in self.rgb_presets.items() if list(bands) == self.rgb_bands), 'Custom')
        self.pn_rgb_preset = pn.widgets.Select(name='Band combination', options=list(self.rgb_presets) + ['Custom'],
                                               value=current)
        self.pn_gamma = pn.widgets.FloatSlider(name='Gamma', value=gamma, start=0.2, end=1.5, step=0.05)
        self.pn_rgb_preset.param.watch(self._on_rgb_preset, 'value')
        for widget in (self.pn_red, self.pn_green, self.pn_blue):
            widget.param.watch(self._on_rgb_band, 'value')

    def _nearest_band(self, wl):
        return self.wls[np.abs(self.wls - wl).argmin()]

    @staticmethod
    def _band_label(wl):
        return '{:.1f}'.format(wl).rstrip('0').rstrip('.') + ' nm'

    def _on_rgb_preset(self, event):
        if event.new in self.rgb_presets:
            self.pn_red.value, self.pn_green.value, self.pn_blue.value = self.rgb_presets[event.new]

    def _on_rgb_band(self, event):
        bands = (self.pn_red.value, self.pn_green.value, self.pn_blue.value)
        if self.rgb_presets.get(self.pn_rgb_preset.value) != bands:
            self.pn_rgb_preset.value = next(
                (name for name, preset in self.rgb_presets.items() if preset == bands), 'Custom')

    def rgb_controls(self):
        """
        Widgets of the RGB composite: band combination (presets), bands of the red, green and blue channels,
        and gamma (exponent applied to Rrs to enhance the dark water pixels), e.g. to place them in another
        layout (see :mod:`grstbx.app`).
        """
        return pn.Column(self.pn_rgb_preset, pn.Row(self.pn_red, self.pn_green, self.pn_blue), self.pn_gamma,
                         sizing_mode='stretch_width')

    def _rgb_bindings(self):
        bands = pn.bind(lambda red, green, blue: (red, green, blue), self.pn_red, self.pn_green, self.pn_blue)
        return {'bands': bands, 'gamma': self.pn_gamma}

    def _rgb_stretch(self, idate, bands=None, gamma=None):
        """Percentiles (2, 98) of the composite, from the coarsest level (one per image and band set)."""
        bands = tuple(self.rgb_bands if bands is None else bands)
        gamma = self.gamma if gamma is None else gamma
        key = (idate, bands, gamma)
        if key not in self._stretch:
            image = self.indexes[idate]
            rgb = self.data.layer(image, list(bands), self.data.nlevels(image) - 1).load()
            rgb = rgb.clip(min=0) ** gamma
            self._stretch[key] = (float(rgb.quantile(0.02)), float(rgb.quantile(0.98)))
        return self._stretch[key]

    has_rgb = True

    def _is_rgb(self, layer):
        return layer == 'RGB'

    def _render(self, date, layer, title, x_range, y_range, width, height, bands=None, gamma=None, contrast=1.,
                brightness=0.):
        if layer != 'RGB':
            return self._image(date, layer, title, x_range, y_range, width, height)

        bands = list(self.rgb_bands if bands is None else bands)
        gamma = self.gamma if gamma is None else gamma
        title = title.replace('RGB', 'RGB {} nm'.format('/'.join('{:.0f}'.format(wl) for wl in bands)))
        x_range, y_range = self._ranges(date, x_range, y_range)
        da = self._read(date, bands, x_range, y_range, width or self.width, height or self.height)
        if da is None:
            return hv.RGB(np.zeros((1, 1, 4)), bounds=(x_range[0], y_range[0], x_range[1], y_range[1]),
                          vdims=['R', 'G', 'B', 'A']).opts(title=title)
        vmin, vmax = self._rgb_stretch(date, bands, gamma)
        da = da.transpose('wl', 'y', 'x')
        if da.y[0] < da.y[-1]:
            da = da.isel(y=slice(None, None, -1))
        values = (da.values.clip(min=0) ** gamma - vmin) / (vmax - vmin)
        values = ((values - 0.5) * contrast + 0.5 + brightness).clip(0, 1)
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
            for ipoint, (x, y) in enumerate(zip(data.get('x', []), data.get('y', []))):
                spectrum = self.data.pixel(self.indexes[date], *self._to_native.transform(x, y))
                if spectrum is None:
                    continue
                curves[ipoint] = hv.Curve((spectrum.wl.values, spectrum.values), 'Wavelength (nm)',
                                          label).opts(color=self.point_color(ipoint))
        if not curves:
            curves[0] = hv.Curve([], 'Wavelength (nm)', label)
        return hv.NdOverlay(curves, kdims='point').opts(
            opts.Curve(width=self.width, height=350, tools=['hover'], line_width=2, show_grid=True),
            opts.NdOverlay(show_legend=False, title='Spectra of the points (full resolution)'))

    def _live_spectrum(self, date, live, x=None, y=None):
        """Full-resolution spectrum of the pixel under the mouse."""
        label = r'Rrs (sr-1)'
        spectrum = None
        if live and x is not None and y is not None:
            spectrum = self.data.pixel(self.indexes[date], *self._to_native.transform(x, y))
        if spectrum is None:
            return hv.Curve([], 'Wavelength (nm)', label)
        return hv.Curve((spectrum.wl.values, spectrum.values), 'Wavelength (nm)', label)

    def visu(self, controls=True):
        """
        Return the panel layout of the viewer.

        :param controls: show the contrast, brightness and RGB composite widgets in the viewer (otherwise they
                         can be placed elsewhere with :meth:`display_controls` and :meth:`rgb_controls`)
        """

        layers = {'RGB': 'RGB'}
        layers.update({'{:.0f}'.format(wl): wl for wl in self.bands})
        pn_band = pn.widgets.RadioButtonGroup(value='RGB', options=layers)
        pn_date = self._date_widget()
        pn_colormap, pn_reverse, cmap = self._colormap_widgets()
        pn_log = pn.widgets.Checkbox(name='Log scale', value=False)
        pn_live = pn.widgets.Checkbox(name='Live spectrum', value=self.live)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0],
                                                      end=self.minmax[1], value=self.minmaxvalues, step=0.0001)
        pn_basemaps = self._basemap_widget()

        def title(date, layer):
            band = 'RGB' if layer == 'RGB' else 'wl = {:.0f} nm'.format(layer)
            return '{}, {}'.format(self.datetimes[date], band)

        map_ = self._map(pn_band, pn_date, cmap, pn_opacity, range_slider, pn_basemaps, title, pn_log)
        spectra = hv.DynamicMap(pn.bind(self._spectra, date=pn_date), streams=[self.poi_stream])
        # spectrum under the mouse (HoloViews throttles the pointer events and keeps the last one)
        pointer = hv.streams.PointerXY(x=None, y=None, source=self.poi_points)
        live = hv.DynamicMap(pn.bind(self._live_spectrum, date=pn_date, live=pn_live),
                             streams=[pointer]).opts(
            opts.Curve(color='black', line_dash='dashed', line_width=1.5, framewise=True))
        spectra = (spectra * live).opts(title='Spectra of the points and under the mouse (full resolution)')

        return pn.Column(
            _to_front(pn.WidgetBox(
                self.title,
                pn.Column(
                    pn.Row('### Band', pn_band, pn_live),
                    pn.Row(
                        pn.Row('### Date', pn_date),
                        pn.Row('#### Basemap', pn_basemaps)
                    ),
                    pn.Row(pn.Row('', range_slider), pn.Row('#### Opacity', pn_opacity)),
                    pn.Row('#### Colormap', pn_colormap, pn.Column(pn_reverse, pn_log)),
                    *([pn.Row('#### RGB', self.pn_rgb_preset, self.pn_red, self.pn_green, self.pn_blue,
                              self.pn_gamma),
                       pn.Row('#### Display', self.pn_contrast, self.pn_brightness)] if controls else [])
                ),
            )),
            pn.pane.HoloViews((map_ * self.poi_points).opts(
                opts.Points(active_tools=['point_draw'], color='color'))),
            pn.pane.HoloViews(spectra)
        )


class ViewParam(_MultiscaleViewer):
    """
    Interactive viewer of L2B images (one 2D map per parameter and date), areas of interest (polygons)
    and points of interest whose full-resolution time series of the parameter are plotted. The time
    series of the pixel under the mouse is also plotted live (dashed line), which can be turned off with
    the 'Live time series' checkbox; the date displayed on the map is marked by a vertical line. The
    time series are displayed beneath the map.

    Only the pixels visible on the screen are read, from the coarsest resolution level that matches
    the zoom (see :class:`ViewSpectral` and :class:`Multiscale`).

    Example
    -------
    >>> viewer = ViewParam(['S2A_L2B_1.zarr', 'S2B_L2B_2.zarr'], reproject=True)
    >>> viewer = ViewParam(datacube, params=['SPM_nechad', 'Chla_OC2nasa'], reproject=True)
    >>> viewer.visu()
    >>> points = viewer.get_points()       # drawn points, EPSG:4326

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
    :param cmap: initial colormap, label of :data:`RRS_COLORMAPS`
                 (e.g. 'Algae, chlorophyll (cmocean algae)'), or any colorcet, cmocean
                 ('cmo.<name>') or matplotlib name
    :param width: width of the map (pixels)
    :param height: height of the map (pixels)
    :param aoi: initial polygons of the areas of interest, returned by :meth:`get_geom` until other
                polygons are drawn (e.g. for non-interactive runs): GeoDataFrame or GeoSeries, shapely
                (Multi)Polygon, WKT string or list of them (EPSG:4326)
    :param poi: initial points of interest, returned by :meth:`get_points` (and plotted) until they are
                moved or deleted: GeoDataFrame or GeoSeries, shapely Point, WKT string, (lon, lat) tuple
                or list of them (EPSG:4326)
    :param title: title of the viewer, in Markdown (e.g. '## EnMAP L2B'), default '## S2 L2B'
    :param live: initial state of the *Live time series* checkbox (reading the time series under the mouse
                 reads one block of each date: untick it for long time series)
    """

    def __init__(self, raster, dates=None,
                 params=None,
                 reproject=False,
                 minmaxvalues=(0, 4),
                 minmax=(0, 10),
                 width=1000,
                 height=700,
                 basemap=DEFAULT_BASEMAP,
                 cmap=DEFAULT_RRS_COLORMAP,
                 aoi=None,
                 poi=None,
                 title='## S2 L2B',
                 live=True):

        super().__init__(raster, dates=dates, reproject=reproject, minmaxvalues=minmaxvalues, minmax=minmax,
                         width=width, height=height, basemap=basemap, cmap=cmap, aoi=aoi,
                         poi=poi)
        self.title = title
        self.live = live

        image = self.data.images[self.indexes[0]][0]
        self.params = params
        if params is None:
            # keep the 2D maps
            self.params = [param for param in image.data_vars
                           if set(image[param].dims) == {'x', 'y'}]

    def _timeseries_curve(self, param, x=None, y=None, color=None):
        """
        Curve and markers of the full-resolution time series of the pixel at (x, y) (CRS of the map);
        empty (but with a time axis, so that the x axis of the plot is a datetime axis) without (x, y).
        """
        series = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        if x is not None and y is not None:
            series = self.data.timeseries(param, *self._to_native.transform(x, y),
                                          indexes=self.indexes).dropna()
        curve = hv.Curve((series.index, series.values), 'Time', param)
        points = hv.Scatter((series.index, series.values), 'Time', param)
        if color is not None:
            curve, points = curve.opts(color=color), points.opts(color=color)
        return curve * points

    def _timeseries(self, data, date, param):
        """Full-resolution time series of the points drawn on the map."""
        curves = {}
        if data:
            for ipoint, (x, y) in enumerate(zip(data.get('x', []), data.get('y', []))):
                curves[ipoint] = self._timeseries_curve(param, x, y, color=self.point_color(ipoint))
        if not curves:
            curves[0] = self._timeseries_curve(param)
        return hv.NdOverlay(curves, kdims='point').opts(show_legend=False) * hv.VLine(self.datetimes[date])

    def _live_timeseries(self, param, live, x=None, y=None):
        """Full-resolution time series of the parameter at the pixel under the mouse."""
        if not live:
            x = y = None
        return self._timeseries_curve(param, x, y)

    def visu(self, controls=True):
        """
        Return the panel layout of the viewer.

        :param controls: show the contrast and brightness widgets in the viewer (otherwise they can be placed
                         elsewhere with :meth:`display_controls`)
        """

        pn_param = pn.widgets.Select(value=self.params[0], options=self.params)
        pn_date = self._date_widget()
        pn_colormap, pn_reverse, cmap = self._colormap_widgets()
        pn_log = pn.widgets.Checkbox(name='Log scale', value=False)
        pn_live = pn.widgets.Checkbox(name='Live time series', value=self.live)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0],
                                                      end=self.minmax[1], value=self.minmaxvalues, step=0.0001)
        pn_basemaps = self._basemap_widget()

        def title(date, param):
            return '{}, {}'.format(self.datetimes[date], param)

        map_ = self._map(pn_param, pn_date, cmap, pn_opacity, range_slider, pn_basemaps, title, pn_log)
        points = hv.DynamicMap(pn.bind(self._timeseries, date=pn_date, param=pn_param),
                               streams=[self.poi_stream])
        # time series under the mouse (HoloViews throttles the pointer events and keeps the last one)
        pointer = hv.streams.PointerXY(x=None, y=None, source=self.poi_points)
        live = hv.DynamicMap(pn.bind(self._live_timeseries, param=pn_param, live=pn_live),
                             streams=[pointer]).opts(
            opts.Curve(color='black', line_dash='dashed', line_width=1.5, framewise=True),
            opts.Scatter(color='black', size=4, tools=['hover'], framewise=True))
        points = points.opts(
            opts.Curve(line_width=1.5, framewise=True),
            opts.Scatter(size=5, tools=['hover'], framewise=True),
            opts.VLine(color='grey', line_dash='dotted', line_width=1.5))
        dates = bokeh.models.DatetimeTickFormatter(days='%Y-%m-%d', months='%Y-%m', years='%Y')
        timeseries = (points * live).opts(
            opts.Overlay(width=self.width, height=350, show_grid=True, framewise=True, xformatter=dates,
                         xrotation=30, title='Time series of the points and under the mouse (full resolution)'))

        return pn.Column(
            _to_front(pn.WidgetBox(
                self.title,
                pn.Column(
                    pn.Row('### Parameter', pn_param, pn_live),
                    pn.Row(
                        pn.Row('### Date', pn_date),
                        pn.Row('#### Basemap', pn_basemaps)
                    ),
                    pn.Row(range_slider, pn.Row('#### Opacity', pn_opacity)),
                    pn.Row('#### Colormap', pn_colormap, pn.Column(pn_reverse, pn_log)),
                    *([pn.Row('#### Display', self.pn_contrast, self.pn_brightness)] if controls else [])
                ),
            )),
            pn.pane.HoloViews((map_ * self.poi_points).opts(
                opts.Points(active_tools=['point_draw'], color='color'))),
            pn.pane.HoloViews(timeseries)
        )
