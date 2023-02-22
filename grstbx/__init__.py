__version__ = '1.0.5'
'''
version 1.0.3: fix for reprojection in any system, 
 specially useful for reprojection in pseudo-mercator EPSG:3857
 for visualization with web mapping tools (e.g., OpenStreetViews, google map)

version 1.0.4: add tools for L2B handling
version 1.0.5: revisit datacube and raster object for multi-tile access

'''

from .driver import l2grs
from .masking import masking
from .utils import *
from .datalake import select_files
