import geopandas
from shapely import wkt

import grstbx

ust = grstbx.utils.spatiotemp()
box = ust.wktbox(7.5, 41.5, width=100, height=100, ellps='WGS84')
bbox = geopandas.GeoSeries.from_wkt([box]).set_crs(epsg=4326)
bbox.to_crs(epsg=4326)
bbox.to_crs(epsg=32631)