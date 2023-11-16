'''
version 1.0.3: fix for reprojection in any system, 
 specially useful for reprojection in pseudo-mercator EPSG:3857
 for visualization with web mapping tools (e.g., OpenStreetViews, google map)

version 1.0.4: add tools for L2B handling
version 1.0.5: revisit datacube and raster object for multi-tile access
version 2.0.0: transition to GRS V2
v2.0.1: fix for gdal projection for accepted dtype, fix for dashboard visu
v2.0.2: small changes for the visu devices
'''

__version__ = '2.0.2'

from .driver import l2grs
from .driver_v1 import l2grs_v1

from .masking import masking
from .utils import *
from .datalake import select_files
from . import visual
