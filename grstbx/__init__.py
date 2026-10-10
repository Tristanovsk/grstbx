'''
version 1.0.3: fix for reprojection in any system, 
 specially useful for reprojection in pseudo-mercator EPSG:3857
 for visualization with web mapping tools (e.g., OpenStreetViews, google map)

version 1.0.4: add tools for L2B handling
version 1.0.5: revisit datacube and raster object for multi-tile access
version 2.0.0: transition to GRS V2
v2.0.1: fix for gdal projection for accepted dtype, fix for dashboard visu
v2.0.2: small changes for the visu devices
v2.0.3: (2026-04-08) add tool option for masking / bitmask flagging
v2.0.4: code optimization and documentation; lazy loading of the visual module
v2.1.0: code optimization and update with pyproject.toml
v2.1.1: open Zarr v3 multiscale (pyramid) images, groups '0', '1', '2', ...
v2.2.0: crop L2A products to an AOI and export the subsets (grstbx.subset); ancillary group of Zarr products;
        map helpers utm_projection and plot_rgb
        packaging: dependency lower bounds, extras (notebook, regrid, docs, all), environment.yml
v2.3.0: multiscale viewers (grstbx.visual): ViewSpectral / ViewParam read only the visible pixels at screen
        resolution from Zarr pyramids (or levels computed on demand), RGB composite, spectra of the drawn
        points, free basemaps (CARTO, OSM, OpenTopoMap, Esri, Stamen via Stadia Maps)
'''

__version__ = '2.3.0'

from .driver import L2grs, open_zarr_image, open_zarr_ancillary, zarr_levels
#from .driver_v1 import l2grs_v1

from .masking import Masking
from .utils import *
from .datalake import SelectFiles
from .subset import open_l2a, aoi_from_box, crop_l2a, export_l2a, export_rrs_geotiff, subset_path, crop_and_export


def __getattr__(name):
    # import the (slow to load) holoviews/datashader-based viewers only when used
    if name == 'visual':
        import importlib
        return importlib.import_module('.visual', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

