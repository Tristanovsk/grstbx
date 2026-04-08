import os
import numpy as np
import pandas as pd
import xarray as xr


class Masking():
    def __init__(self, product, flag_ID='flags', names_='flag_names',
                 description_='flag_descriptions',
                 ):
        self.product = product

        self.flag_ID = flag_ID
        self.names_ = names_
        self.description_ = description_

    def print_info(self):
        self.get_flags()
        return self.dflags

    def get_flags(self, ):

        pflags = self.product[self.flag_ID]
        #names = []
        #for flag_name in pflags.attrs[self.names_].split(' '):
        #    names.append(flag_name)
        names = pflags.attrs[self.names_]
        # construct dataframe:
        dflags = pd.DataFrame({'name': names})
        dflags['description'] = pflags.attrs[self.description_]#.split('\t')
        dflags['bit'] = dflags.index
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

    @staticmethod
    def add_flag(flags,
                 boolean_cond,
                 name,
                 bitmask,
                 description=''):
        xr.set_options(keep_attrs=True)
        flags = flags + (boolean_cond << bitmask)

        # add name and description
        flags.attrs['flag_descriptions'][bitmask] = description
        flags.attrs['flag_names'][bitmask] = name
        return flags

    def compute_mask_value(self, **flags):

        mask = 0
        for flag_name, flag_ref in flags.items():
            bit, bit_val = self.dflags.loc[flag_name, ['bit', 'value']]
            #print(flag_name, flag_ref, bit)
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

    @staticmethod
    def create_mask(flags,
                    tomask=[0, 2],
                    tokeep=[3],
                    mask_name="mask",
                    _type=np.uint8
                    ):
        '''
        Create binary mask from bitmask flags, with selection of bitmask to mask or to keep (by bit number).
        The masking convention is: good pixels for mask == 0, bad pixels when mask == 1

        :param flags: xarray dataarray with bitmask flags
        :param tomask: array of bitmask flags used to mask
        :param tokeep: array of bitmask flags for which pixels are kept (= good quality)
        :param mask_name: name of the output mask
        :param _type: type of the array (uint8 is recommended)
        :return: mask

        Example of output mask

        >>> mask = create_mask(raster.flags,
        ...                    tomask = [0,2,11],
        ...                    tokeep = [3],
        ...                    mask_name="mask_from_flags" )
        <xarray.DataArray>
        'mask_from_flags'
        y: 5490x: 5490
        array([[1, 1, 1, ..., 1, 1, 1],
               [1, 1, 1, ..., 1, 1, 1],
               [1, 1, 1, ..., 1, 1, 1],
               ...,
               [0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 1, 1]], dtype=uint8)
        Coordinates:
            x           (x) float64 6e+05 6e+05 ... 7.098e+05 7.098e+05
            y           (y) float64 4.9e+06 4.9e+06 ... 4.79e+06
            spatial_ref () int64 0
            time        () datetime64[ns] 2021-05-12T10:40:21
            band        () int64 1
        Indexes: (2)
        Attributes:
        long_name:   binary mask from flags
        description: good pixels for mask == 0, bad pixels when mask == 1

        '''

        mask = xr.zeros_like(flags, dtype=_type)

        flag_value_tomask = 0
        flag_value_tokeep = 0

        if len(tomask) > 0:
            for bitnum in tomask:
                flag_value_tomask += 1 << bitnum

        if len(tokeep) > 0:
            for bitnum in tokeep:
                flag_value_tokeep += 1 << bitnum

        if (len(tokeep) > 0) & (len(tomask) > 0):
            mask = (((flags & flag_value_tomask) != 0) | ((flags & flag_value_tokeep) == 0)).astype(_type)
        elif (len(tokeep) > 0) | (len(tomask) > 0):
            if len(tokeep) > 0:
                mask = ((flags & flag_value_tokeep) == 0)
            else:
                mask = ((flags & flag_value_tomask) != 0)
        mask.attrs["long_name"] = "binary mask from flags"
        mask.attrs["description"] = "good pixels for mask == 0, bad pixels when mask == 1"
        mask.name = mask_name
        return mask
