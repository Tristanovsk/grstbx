import os
import glob
import pandas as pd
import panel as pn

opj = os.path.join


class SelectFiles():
    def __init__(self, root=None):
        if root is None:
            self.root = '/datalake/watcal'
        else:
            self.root = root

    def select_pattern(self, pattern='*.nc'):
        list_ = pd.DataFrame(glob.glob(opj(self.root, pattern)))
        if len(list_) == 0:
            print("your path:", opj(self.root, pattern))
            print("wrong path, no data available; try again!")
            return
        basenames = list_[0].str.split('/', expand=True).iloc[:, -1]
        basenames = basenames.str.split('_', expand=True)
        file_list = basenames.iloc[..., [0, 1, 2, 5, 7, -1]].copy()
        file_list.columns = ['satellite', 'level', 'date', 'tile', 'cloud_coverage', 'version']
        file_list['version'] = file_list['version'].str.split('.', expand=True).values[:, 0]
        file_list['cloud_coverage'] = file_list['cloud_coverage'].str.replace('cc', '').astype(float)
        file_list['abspath'] = list_
        file_list['date'] = pd.to_datetime(file_list['date'])
        self.file_list = file_list.set_index('date').sort_index()

    def list_folder(self,  pattern='*.nc'):
        '''


        :param pattern: regex pattern to pre-select your files (default: '*.nc')
        :return:
        '''

        datadir = opj(self.root,  pattern)
        list_ = glob.glob(datadir)
        if len(list_) == 0:

            print("wrong path, no data available; try again!")
            return

        basenames = pd.Series([os.path.basename(p) for p in list_])  # ,columns=['basename']) #basenames
        file_list = basenames.str.split('_', expand=True).iloc[..., [0, 1, 2, 5, 7, -1]].copy()
        file_list.columns = ['satellite', 'level', 'date', 'tile', 'cloud_coverage', 'version']
        file_list['version'] = file_list['version'].str.split('.', expand=True).values[:, 0]
        file_list['cloud_coverage'] = file_list['cloud_coverage'].str.replace('cc', '').astype(float)
        file_list['abspath'] = list_
        file_list['date'] = pd.to_datetime(file_list['date'])
        # file_list = list_[0].str.split('/', expand=True).iloc[:, nb_rep:]
        # file_list.columns = ['product', 'tile', 'year', 'month', 'day', 'image']
        # file_list['basenames'] = [os.path.basename(p) for p in file_list['abspath']]
        # file_list['date'] = pd.to_datetime(file_list[['year', 'month', 'day']])
        self.file_list = file_list.set_index('date').sort_index()

    def list_tile(self, product='S2-L2GRS', tile='31TEJ', pattern='*.nc'):
        '''

        :param product: desired product type, usually corresponds
                        to the name of the folder containing the data (default: 'S2-L2GRS)
        :param tile: tile name
        :param pattern: regex pattern to pre-select your files (default: '*.nc')
        :return:
        '''

        datadir = opj(self.root, product, tile, '*', '*', '*', pattern)
        list_ = glob.glob(datadir)
        if len(list_) == 0:
            print("your path:", opj(self.root, product, tile, '*', '*', '*', pattern))
            print("wrong path, no data available; try again!")
            return

        basenames = pd.Series([os.path.basename(p) for p in list_])  # ,columns=['basename']) #basenames
        file_list = basenames.str.split('_', expand=True).iloc[..., [0, 1, 2, 5, 7, -1]].copy()
        file_list.columns = ['satellite', 'level', 'date', 'tile', 'cloud_coverage', 'version']
        file_list['version'] = file_list['version'].str.split('.', expand=True).values[:, 0]
        file_list['cloud_coverage'] = file_list['cloud_coverage'].str.replace('cc', '').astype(float)
        file_list['abspath'] = list_
        file_list['date'] = pd.to_datetime(file_list['date'])
        # file_list = list_[0].str.split('/', expand=True).iloc[:, nb_rep:]
        # file_list.columns = ['product', 'tile', 'year', 'month', 'day', 'image']
        # file_list['basenames'] = [os.path.basename(p) for p in file_list['abspath']]
        # file_list['date'] = pd.to_datetime(file_list[['year', 'month', 'day']])
        self.file_list = file_list.set_index('date').sort_index()

    def select_dates(self):

        values = (self.file_list.index[0], self.file_list.index[-1])

        return pn.widgets.DatetimeRangePicker(name='Datetime Range Picker', value=values)

    def list_file_path(self, date_startend=('2015-01-01', '2023-12-31'), cc_max=None):
        startdate, enddate = date_startend
        self.file_list = self.file_list[startdate:enddate]
        if cc_max:
            self.file_list = self.file_list[self.file_list.cloud_coverage < cc_max]
        self.files = self.file_list.abspath.values
        return self.files
