import os
import numpy as np
import pandas as pd
import xarray as xr


class masking():
    def __init__(self, product ,flag_ID='flags', names_='flag_meanings',
                  description_='flag_descriptions',
                  mask_binary='flag_masks'):
        self.product = product

        self.flag_ID=flag_ID
        self.names_=names_
        self.description_=description_
        self.mask_binary=mask_binary
        self.get_flags()

    def print_info(self):
        return self.dflags

    def get_flags(self, ):

        pflags = self.product[self.flag_ID]
        names = []
        for flag_name in pflags.attrs[self.names_].split(' '):
            names.append(flag_name)

        # construct dataframe:
        dflags = pd.DataFrame({'name': names})
        dflags['description'] = pflags.attrs[self.description_].split('\t')
        dflags['value'] = pflags.attrs[self.mask_binary]
        dflags.sort_values('value',inplace=True)
        dflags['bit']=dflags.index
        self.dflags = dflags.set_index('name')
        self.pflags = pflags

    @staticmethod
    def bitmask(mask, bitval, value):
        """

        """

        if value:
            mask |= bitval
        else:
            mask &= (~bitval)
        return mask

    def compute_mask_value(self, **flags):

        mask = 0
        for flag_name, flag_ref in flags.items():
            bit, bit_val = self.dflags.loc[flag_name, ['bit','value']]
            print(flag_name, flag_ref, bit)
            mask = self.bitmask(mask, bit_val, True)
            value = self.bitmask(mask, bit_val, flag_ref)
            self.mask = mask
            self.value = value
        return mask, value

    def get_mask(self, **flags):
        """
        Returns boolean xarray computed from **flags
        Example:
        masking_ = masking(product)
        mask_ = masking_.make_mask(MG2_Water_Mask=False,negative=False,nodata=True)

        :param flags: list of boolean flags
        :return: boolean xarray.DataArray
        """

        mask, value = self.compute_mask_value(**flags)


        return self.product[self.flag_ID] & mask == value
