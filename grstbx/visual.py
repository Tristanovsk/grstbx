import glob
import os
import numpy as np
import pandas as pd

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
import hvplot.xarray

import holoviews as hv
import holoviews.operation.datashader as hd
from holoviews.operation.datashader import rasterize, shade, spread
from holoviews.element.tiles import EsriImagery
from holoviews.element import tiles as hvts

hv.extension('bokeh')

import datashader as ds
import datashader.transfer_functions as tf

import colorcet as cc
import panel as pn
import panel.widgets as pnw
import param as pm

from collections import OrderedDict as odict


class image_viewer():

    def Rrs_date(self, raster):
        ps = {k: p for k, p in cc.palette_n.items()}
        ps = odict([(n, cc.palette[n]) for n in ['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc']])
        iwls = {wl: iwl for iwl, wl in enumerate(raster.wl.values)}
        dates = {str(date): idate for idate, date in enumerate(raster.time.values)}
        map_tiles = EsriImagery().opts(alpha=0.65, bgcolor='black')
        maps = ['EsriImagery', 'EsriUSATopo', 'EsriTerrain', 'StamenWatercolor', 'StamenTonerBackground']
        bases = odict(
            [(name, None) if name == 'None' else (name, getattr(hvts, name)().relabel(name)) for name in maps])
        gopts = hv.opts.Tiles(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

        class Viewer(pm.Parameterized):
            cmap = pm.Selector(ps, default=ps['kbc'])
            wavelength = pm.Selector(iwls, default=2)
            date = pm.Selector(dates)
            basemap = pm.Selector(bases)

            @pm.depends('wavelength', 'date', 'cmap')
            def select_band(self):
                d_ = raster.isel(wl=self.wavelength, time=self.date)
                title = 'Band at {:.2f}'.format(d_.wl.values) + ' nm'
                hv_dataset_large = hv.Dataset(d_, kdims=['x', 'y'])
                hv_image_large = hv.Image(hv_dataset_large, ['x', 'y']).opts(width=900, height=600)

                return map_tiles * hd.regrid(hv_image_large).opts(tools=['hover'], active_tools=['wheel_zoom'],
                                                                  cmap=self.cmap,
                                                                  colorbar=True, colorbar_position='bottom',
                                                                  title=title, clim=(0, None), cnorm='eq_hist')  # )

            @pm.depends('basemap')
            def tiles(self):
                if self.basemap is None:
                    return
                return self.basemap.opts(gopts).opts(alpha=0.5)

            def map_band(self):  # iwl=2,cmap=ps['kbc']):
                ropts = dict(cmap=self.cmap, )
                return hv.DynamicMap(self.tiles) * self.select_band()  # .opts(**ropts) 

        viewer = Viewer()

        return pn.Row(pn.Param(viewer.param), viewer.map_band)  # iwl=2, cmap=ps))

    def param_date(self, raster):

        ps = {k: p for k, p in cc.palette_n.items()}
        ps = odict([(n, cc.palette[n]) for n in ['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc']])
        dates = {str(date): idate for idate, date in enumerate(raster.time.values)}
        maps = ['EsriImagery', 'EsriUSATopo', 'EsriTerrain', 'StamenWatercolor', 'StamenTonerBackground']
        bases = odict(
            [(name, None) if name == 'None' else (name, getattr(hvts, name)().relabel(name)) for name in maps])
        gopts = hv.opts.Tiles(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)

        class Viewer(pm.Parameterized):
            cmap = pm.Selector(ps, default=ps['kbc'])
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
