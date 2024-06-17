import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')

#import hvplot.xarray
import holoviews as hv
import holoviews.operation.datashader as hd
from holoviews.operation.datashader import rasterize, shade, spread
from holoviews.element.tiles import EsriImagery
from holoviews.element import tiles as hvts
from holoviews import opts



import datashader as ds
import datashader.transfer_functions as tf
import bokeh
import colorcet as cc
import panel as pn
import panel.widgets as pnw
import param as pm
from shapely.geometry import Polygon
from collections import OrderedDict as odict

hv.extension('bokeh')


class ImageViewer():

    def Rrs_date(self, raster, third_dim='wl', param='Rrs', Rrs_unit=True):

        param_label = r'$R_{rs}$'
        if not Rrs_unit:
            param_label = r'$rho_w$'

        ps = {k: p for k, p in cc.palette_n.items()}
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
                print(data)
                ds_ = self.ds_  # hv.Dataset(raster.isel(time=self.date,drop=True))
                curves = {}
                data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
                i = 0
                for x0, x1, y0, y1 in data:
                    if i == 0:
                        self.extracted_data = raster.sel(x=slice(x0, x1), y=slice(y0, y1))
                    selection = ds_.select(x=(x0, x1), y=(y0, y1))

                    mean = selection.aggregate(third_dim, np.nanmean).data
                    # return hv.NdOverlay({0: hv.Curve([],'OK', param_label)})
                    if np.isnan(mean[param][0]):
                        continue
                    std = selection.aggregate(third_dim, np.nanstd).data

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

            def map_band(self):  # iwl=2,cmap=ps['kbc']):
                self.extract_ds_by_date()
                ropts = dict(cmap=self.cmap, )
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

        cmap_ = cmap
        ps = {k: p for k, p in cc.palette_n.items()}
        ps = odict([(n, cc.palette[n]) for n in ['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc']])
        dates = {str(date): idate for idate, date in enumerate(raster.time.values)}
        maps = ['EsriImagery', 'EsriUSATopo', 'EsriTerrain', 'StamenWatercolor', 'StamenTonerBackground']
        bases = odict(
            [(name, None) if name == 'None' else (name, getattr(hvts, name)().relabel(name)) for name in maps])
        gopts = hv.opts.Tiles(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

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
                ropts = dict(cmap=self.cmap, )
                return hv.DynamicMap(self.tiles) * self.select_date()  # .opts(**ropts)

        viewer = Viewer()

        return pn.Row(pn.Param(viewer.param), viewer.map_band)


class Utils():

    @staticmethod
    def get_geom(aoi_stream, crs=4326):
        geom = aoi_stream.data
        ys, xs = geom['ys'][-1], geom['xs'][-1]
        polygon_geom = Polygon(zip(xs, ys))
        polygon = gpd.GeoDataFrame(index=[0], crs=3857, geometry=[polygon_geom])
        return polygon.to_crs(crs)

    @staticmethod
    def custom_hover():
        formatter_code = """
          var digits = 4;
          var projections = Bokeh.require("core/util/projections");
          var x = special_vars.x; var y = special_vars.y;
          var coords = projections.wgs84_mercator.invert(x, y);
          return "" + (Math.round(coords[%d] * 10**digits) / 10**digits).toFixed(digits)+ "";
        """
        formatter_code_x, formatter_code_y = formatter_code % 0, formatter_code % 1
        custom_tooltips = [('Lon', '@x{custom}'), ('Lat', '@y{custom}'), ('Value', '@image{0.0000}')]
        custom_formatters = {
            '@x': bokeh.models.CustomJSHover(code=formatter_code_x),
            '@y': bokeh.models.CustomJSHover(code=formatter_code_y)
        }
        return bokeh.models.HoverTool(tooltips=custom_tooltips, formatters=custom_formatters)


class ViewSpectral(Utils):
    def __init__(self, raster, dates=None,
                 bands=None,
                 reproject=False,
                 minmaxvalues=(0, 0.02),
                 minmax=(0, 0.06)
                 ):

        # layout settings
        self.title = '## S2 L2A'
        self.width, self.height = 1200, 800
        self.key_dimensions = ['x', 'y']
        self.minmaxvalues = minmaxvalues
        self.minmax = minmax
        self.colormaps = ['CET_D13', 'bky', 'CET_D1A', 'CET_CBL2', 'CET_L10', 'CET_C6s',
                          'kbc', 'blues_r', 'kb', 'rainbow', 'fire', 'kgy', 'bjy', 'gray']

        # check if single date, if so push time as dimension to be compliant with multidates
        if not 'time' in raster.dims:
            raster = raster.expand_dims('time')

        # variables settings
        self.dates = dates
        if dates == None:
            self.dates = raster.time.dt.date.values
        self.bands = bands
        if bands == None:
            self.bands = raster.wl.values

        # load raster
        self.raster = raster
        self.dataarrays = {}

        times = raster.time.values
        if not hasattr(times, "__len__"):
            times = [times]
        for itime, time in enumerate(times):
            raster_ = raster.sel(time=time)
            for iband, band in enumerate(self.bands):
                if reproject:
                    self.dataarrays[itime, iband] = raster_.sel(wl=band).rio.reproject(3857, nodata=np.nan)
                else:
                    self.dataarrays[itime, iband] = raster_.sel(wl=band)


        # declare streaming object to get Area of Interest (AOI), crs=crs.epsg(3857)
        self.aoi_polygons = hv.Polygons([]).opts(opts.Polygons(
            fill_alpha=0.3, fill_color='white',
            line_width=1.2))  ##, active_tools=['poly_draw']))#.opts(crs.GOOGLE_MERCATOR)
        self.aoi_stream = hv.streams.PolyDraw(
            source=self.aoi_polygons, drag=True)  # , num_objects=1)#5,styles={'fill_color': aoi_colours})
        self.edit_stream = hv.streams.PolyEdit(source=self.aoi_polygons, vertex_style={'color': 'red'})


    def visu(self):

        # get data
        dates = self.dates
        bands = self.bands

        # set visualization options
        hv.opts.defaults(
            hv.opts.Image(height=self.height, width=self.width,
                          colorbar=True, tools=[self.custom_hover()], active_tools=['wheel_zoom'],
                          clipping_colors={'NaN': '#00000000'}),
            hv.opts.Tiles(active_tools=['wheel_zoom'])
        )
        gopts = hv.opts.Tiles(xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

        # set images to show
        titles, images = {}, {}
        for idate, date in enumerate(dates):
            for iband, band in enumerate(bands):
                titles[date, iband] = str(date) + ', wl = {:.2f} nm '.format(band)
                datasets = hv.Dataset(self.dataarrays[idate, iband].squeeze(), kdims=self.key_dimensions)
                images[date, iband] = hv.Image(datasets).opts(gopts)

        # set map/color collections
        bases = [name for name, ts in hv.element.tiles.tile_sources.items()]
        pn_band = pn.widgets.RadioButtonGroup(value=2, options=list(range(len(bands))))
        pn_colormap = pn.widgets.Select(value='CET_D13',
                                        options=self.colormaps)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0], end=self.minmax[1],
                                              value=self.minmaxvalues, step=0.0001)
        pn_basemaps = pn.widgets.Select(value=bases[0], options=bases)
        pn_date = pn.widgets.DatePicker(value=dates[0], start=dates[0],
                                        enabled_dates=dates.tolist())  # .date, end=dates[-1],value=dates[0])

        @pn.depends(
            pn_date_value=pn_date.param.value,
            pn_band_value=pn_band.param.value,
            pn_colormap_value=pn_colormap.param.value,
            pn_opacity_value=pn_opacity.param.value,
            range_slider_value=range_slider.param.value

        )
        def load_map(pn_date_value, pn_band_value,
                     pn_colormap_value, pn_opacity_value, range_slider_value):
            image = images[pn_date_value, pn_band_value]
            used_colormap = cc.cm[pn_colormap_value]
            image.opts(cmap=used_colormap, alpha=pn_opacity_value, clim=range_slider_value,
                       title=titles[pn_date_value, pn_band_value])

            return image

        @pn.depends(
            basemap_value=pn_basemaps.param.value)
        def load_tiles(basemap_value):
            tiles = hv.element.tiles.tile_sources[basemap_value]()
            return tiles.options(height=self.height, width=self.width).opts(gopts)

        dynmap = hd.regrid(hv.DynamicMap(load_map))
        combined = (hv.DynamicMap(
            load_tiles) * dynmap * self.aoi_polygons)  # .opts(active_tools=['wheel_zoom', 'poly_draw'])

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
            combined)
        )


class ViewParam(Utils):
    def __init__(self, raster, dates=None,
                 params=None,
                 reproject=False,
                 minmaxvalues=(0, 4),
                 minmax=(0, 10)
                 ):

        # layout settings
        self.title = '## S2 L2B'
        self.width, self.height = 1200, 800
        self.key_dimensions = ['x', 'y']
        self.minmaxvalues = minmaxvalues
        self.minmax = minmax
        self.colormaps = ['CET_D13', 'bky', 'CET_D1A', 'CET_CBL2', 'CET_L10', 'CET_C6s',
                          'kbc', 'blues_r', 'kb', 'rainbow', 'fire', 'kgy', 'bjy', 'gray']
        # check if single date, if so push time as dimension to be compliant with multidates
        if not 'time' in raster.dims:
            raster = raster.expand_dims('time')

        # variables settings
        self.dates = dates
        if dates == None:
            self.dates = raster.time.dt.date.values
            self.datetimes = raster.time.dt.strftime('%Y-%m-%d %H:%M:%S').values  # dt.date.values

        self.params = params
        if params == None:
            self.params = list()
            # Clean up: param to be removed
            self.params= []
            for param in raster.data_vars:
                if len(raster[param].shape)>3:
                    continue
                if ('x' in raster[param].dims) and ('y' in raster[param].dims):
                    self.params.append(param)
            # for to_be_removed in ['crs', 'metadata']:
            #     if to_be_removed in self.params:
            #         self.params.remove(to_be_removed)

        # load raster
        self.raster = raster
        self.dataarrays = {}

        times = raster.time.values
        if not hasattr(times, "__len__"):
            times = [times]
        for itime, time in enumerate(raster.time.values):
            raster_ = raster.sel(time=time)
            for iparam, param in enumerate(self.params):
                if reproject:
                    self.dataarrays[itime, param] = raster_[param].astype(np.float32).rio.reproject(3857, nodata=np.nan)
                else:
                    self.dataarrays[itime, param] = raster_[param]

        # declare streaming object to get Area of Interest (AOI)
        self.aoi_polygons = hv.Polygons([]).opts(opts.Polygons(
            fill_alpha=0.3, fill_color='white', line_width=1.2))
        self.aoi_stream = hv.streams.PolyDraw(source=self.aoi_polygons)
        self.edit_stream = hv.streams.PolyEdit(source=self.aoi_polygons, vertex_style={'color': 'red'})
        # set visualization options
        hv.opts.defaults(
            hv.opts.Image(height=self.height, width=self.width,
                          colorbar=True, tools=[self.custom_hover(), 'box_select'], active_tools=['wheel_zoom'],
                          clipping_colors={'NaN': '#00000000'}),
            hv.opts.Tiles(active_tools=['wheel_zoom'])
        )


    def visu(self):

        # get data
        dates = self.dates
        params = self.params

        gopts = hv.opts.Tiles(xaxis=None, yaxis=None, bgcolor='black', show_grid=False)
        # set images to show
        titles, images = {}, {}
        for idate, date in enumerate(dates):
            for iparam, param in enumerate(params):
                titles[date, param] = str(date) + ', ' + param
                datasets = hv.Dataset(self.dataarrays[idate, param].squeeze(), kdims=self.key_dimensions)
                images[date, param] = hv.Image(datasets).opts(gopts)

        # set map/color collections
        bases = [name for name, ts in hv.element.tiles.tile_sources.items()]
        pn_param = pn.widgets.Select(value=params[0], options=params)
        pn_colormap = pn.widgets.Select(value='CET_D13',
                                        options=self.colormaps)
        pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.95, start=0, end=1, step=0.05)
        range_slider = pn.widgets.EditableRangeSlider(name='Range Slider', start=self.minmax[0], end=self.minmax[1],
                                              value=self.minmaxvalues, step=0.0001)
        pn_basemaps = pn.widgets.Select(value=bases[0], options=bases)
        pn_date = pn.widgets.DatePicker(value=dates[0], start=dates[0],
                                        enabled_dates=dates.tolist())  # .date, end=dates[-1],value=dates[0])

        @pn.depends(
            pn_date_value=pn_date.param.value,
            pn_param_value=pn_param.param.value,
            pn_colormap_value=pn_colormap.param.value,
            pn_opacity_value=pn_opacity.param.value,
            range_slider_value=range_slider.param.value

        )
        def load_map(pn_date_value, pn_param_value,
                     pn_colormap_value, pn_opacity_value, range_slider_value):
            image = images[pn_date_value, pn_param_value]
            used_colormap = cc.cm[pn_colormap_value]
            image.opts(cmap=used_colormap, alpha=pn_opacity_value, clim=range_slider_value,
                       title=titles[pn_date_value, pn_param_value])
            return image

        @pn.depends(
            basemap_value=pn_basemaps.param.value)
        def load_tiles(basemap_value):
            tiles = hv.element.tiles.tile_sources[basemap_value]()
            return tiles.options(height=self.height, width=self.width).opts(gopts)

        dynmap = hd.regrid(hv.DynamicMap(load_map))
        combined = (hv.DynamicMap(
            load_tiles) * dynmap * self.aoi_polygons)  # .opts(active_tools=['wheel_zoom', 'poly_draw'])

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
            combined)
        )
