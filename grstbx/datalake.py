import os
import glob
import pandas as pd
import panel as pn

opj = os.path.join


class select_files():
    def __init__(self):
        self.root = '/datalake/watcal'

    def list_tile(self, product='S2-L2GRS', tile='31TEJ', pattern='*.nc'):
        '''

        :param product: desired product type, usually corresponds
                        to the name of the folder containing the data (default: 'S2-L2GRS)
        :param tile: tile name
        :param pattern: regex pattern to pre-select your files (default: '*.nc')
        :return:
        '''
        nb_rep = len(self.root.split('/'))
        list_ = pd.DataFrame( )

        list = list_[0].str.split('/', expand=True).iloc[:, nb_rep:]
        list.columns = ['product', 'tile', 'year', 'month', 'day', 'image']
        list['abspath'] = list_
        list['date'] = pd.to_datetime(list[['year', 'month', 'day']])
        self.list = list.set_index('date').sort_index()

    def select_dates(self):

        values = (self.list.index[0],self.list.index[-1])

        return pn.widgets.DatetimeRangePicker(name='Datetime Range Picker', value=values)

    def list_file_path(self,date_startend):
        startdate,enddate = date_startend
        self.files=self.list[startdate:enddate].abspath.values
        return self.files




