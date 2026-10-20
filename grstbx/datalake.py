"""
Tools to list GRS products stored on disk and select them by date and cloud coverage.

File names are expected to follow the GRS convention, i.e. underscore-separated
fields where field 0 is the satellite, 1 the processing level, 2 the date,
5 the tile, 7 the cloud coverage ('ccXX') and the last field the version.
Products with STAC Items (local or in a STAC API) are selected from their Items with
:meth:`SelectFiles.search_stac` instead.
"""

import os
import glob
import pandas as pd

opj = os.path.join


class SelectFiles():
    """
    List GRS products from a data lake.

    Example
    -------
    >>> sf = SelectFiles('/datalake/watcal')
    >>> sf.list_tile(product='S2-L2GRS', tile='31TEJ')
    >>> files = sf.list_file_path(('2022-01-01', '2022-12-31'), cc_max=50)

    Products with STAC Items (e.g. the L2B products of GRSl2bgen ``--stac``, or a STAC API):

    >>> sf = SelectFiles('/data/satellite/Sentinel-2/L2B')
    >>> sf.search_stac(tile='31TFJ', bbox=(4.4, 43.4, 4.7, 43.6), cc_max=20)
    >>> files = sf.list_file_path(('2021-01-01', '2021-12-31'))

    :param root: root directory of the data lake (default: '/datalake/watcal')
    """

    def __init__(self, root=None):
        if root is None:
            self.root = '/datalake/watcal'
        else:
            self.root = root

    @staticmethod
    def _parse_file_list(paths):
        """
        Parse GRS file names into a pandas.DataFrame indexed (and sorted) by date
        with columns: satellite, level, tile, cloud_coverage, version, abspath.
        """
        basenames = pd.Series([os.path.basename(p) for p in paths])
        file_list = basenames.str.split('_', expand=True).iloc[:, [0, 1, 2, 5, 7, -1]].copy()
        file_list.columns = ['satellite', 'level', 'date', 'tile', 'cloud_coverage', 'version']
        file_list['version'] = file_list['version'].str.split('.').str[0]
        file_list['cloud_coverage'] = file_list['cloud_coverage'].str.replace('cc', '').astype(float)
        file_list['abspath'] = list(paths)
        file_list['date'] = pd.to_datetime(file_list['date'])
        return file_list.set_index('date').sort_index()

    def _glob(self, datadir):
        """Glob ``datadir`` and set ``self.file_list``; return False if nothing found."""
        list_ = glob.glob(datadir)
        if len(list_) == 0:
            print("your path:", datadir)
            print("wrong path, no data available; try again!")
            return False
        self.file_list = self._parse_file_list(list_)
        return True

    def select_pattern(self, pattern='*.nc'):
        """
        List files matching ``root/pattern`` into ``self.file_list``.

        :param pattern: glob pattern relative to root (default: ``'*.nc'``)
        """
        self._glob(opj(self.root, pattern))

    def list_folder(self, pattern='*.nc'):
        '''
        List files of the root folder into ``self.file_list``.

        :param pattern: glob pattern to pre-select your files (default: ``'*.nc'``)
        '''
        self._glob(opj(self.root, pattern))

    def list_tile(self, product='S2-L2GRS', tile='31TEJ', pattern='*.nc'):
        '''
        List files of a tile stored as ``root/product/tile/YYYY/MM/DD/pattern``
        into ``self.file_list``.

        :param product: desired product type, usually corresponds
                        to the name of the folder containing the data (default: 'S2-L2GRS')
        :param tile: tile name
        :param pattern: glob pattern to pre-select your files (default: ``'*.nc'``)
        '''
        self._glob(opj(self.root, product, tile, '*', '*', '*', pattern))

    def search_stac(self, source=None, tile=None, bbox=None, intersects=None, date_startend=None, cc_max=None,
                    version=None, collections=None, limit=1000):
        """
        List the products of a STAC catalogue into ``self.file_list`` (see
        :func:`grstbx.app.search_items`): local STAC Items below a folder (default: ``root``), Item files
        or glob patterns, or a STAC API (URL).

        The table has the columns of the file names listing (index ``date``: acquisition time,
        ``satellite``, ``level``, ``tile``, ``cloud_coverage`` (``eo:cloud_cover``, %), ``version`` (GRS
        version), ``abspath``: data asset of the Item, a local path or a URL) and ``item`` (Item file or
        URL), so that :meth:`list_file_path` selects them the same way.

        :param source: folder, Item file or glob pattern, or STAC API URL (default: ``root``)
        :param tile: tile (e.g. ``'31TFJ'``)
        :param bbox: area (lon_min, lat_min, lon_max, lat_max), deg
        :param intersects: area as a shapely geometry or GeoJSON dict (lon/lat), instead of ``bbox``
        :param date_startend: (start, end) acquisition dates, inclusive
        :param cc_max: maximum cloud coverage (%, inclusive); the Items without cloud cover are kept
        :param version: GRS version(s) to keep, e.g. ``'3.1.0'`` or ``'V3.1.0'``
        :param collections: STAC API: collection id or list of ids
        :param limit: maximum number of Items read from a STAC API
        :return: ``self.file_list`` (pandas.DataFrame); empty if nothing matches
        """
        from .app import search_items

        start, end = date_startend if date_startend is not None else (None, None)
        found = search_items(self.root if source is None else source, bbox=bbox, intersects=intersects,
                             start=start, end=end, max_cloud_cover=cc_max, tile=tile,
                             collections=collections, version=version, limit=limit)
        if found.empty:
            print('no STAC Item matches the search')
        file_list = pd.DataFrame({'date': pd.to_datetime(found.time).values,
                                  'satellite': found.satellite.values,
                                  'level': found.level.values,
                                  'tile': found.tile.values,
                                  'cloud_coverage': pd.to_numeric(found.cloud_cover, errors='coerce').values,
                                  'version': found.version.values,
                                  'abspath': found.path.values,
                                  'item': found.item.values})
        self.file_list = file_list.set_index('date').sort_index()
        return self.file_list

    def select_dates(self):
        """Return a panel DatetimeRangePicker spanning the listed dates."""
        import panel as pn

        values = (self.file_list.index[0], self.file_list.index[-1])

        return pn.widgets.DatetimeRangePicker(name='Datetime Range Picker', value=values)

    def list_file_path(self, date_startend=('2015-01-01', '2023-12-31'), cc_max=None):
        """
        Filter ``self.file_list`` by date range and cloud coverage (in place).

        :param date_startend: (start, end) dates, inclusive
        :param cc_max: maximum cloud coverage (strict), no filtering if None
        :return: array of absolute paths of the selected files
        """
        startdate, enddate = date_startend
        self.file_list = self.file_list[startdate:enddate]
        if cc_max:
            self.file_list = self.file_list[self.file_list.cloud_coverage < cc_max]
        self.files = self.file_list.abspath.values
        return self.files
