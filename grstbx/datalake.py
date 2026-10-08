"""
Tools to list GRS products stored on disk and select them by date and cloud coverage.

File names are expected to follow the GRS convention, i.e. underscore-separated
fields where field 0 is the satellite, 1 the processing level, 2 the date,
5 the tile, 7 the cloud coverage ('ccXX') and the last field the version.
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
