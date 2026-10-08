"""
Bitmask handling for GRS products.

GRS stores pixel classification as a single integer ``flags`` raster in which
bit ``i`` is set when condition ``flag_names[i]`` holds. The names and
descriptions of each bit are stored in the attributes of the ``flags``
variable. This module decodes those bits into boolean masks.
"""

import numpy as np
import pandas as pd
import xarray as xr


class Masking():
    """
    Decode the bitmask ``flags`` variable of a GRS product.

    Example
    -------
    >>> masking_ = Masking(product)
    >>> masking_.print_info()                       # table of available flags
    >>> water = masking_.get_mask(ndwi=False, hicld=False)

    :param product: xarray.Dataset holding the bitmask variable
    :param flag_ID: name of the bitmask variable (default: 'flags')
    :param names\\_: attribute of the bitmask variable listing the flag names
    :param description\\_: attribute of the bitmask variable listing the flag descriptions
    """

    def __init__(self, product, flag_ID='flags', names_='flag_names',
                 description_='flag_descriptions',
                 ):
        self.product = product

        self.flag_ID = flag_ID
        self.names_ = names_
        self.description_ = description_

    def print_info(self):
        """Return a pandas.DataFrame describing each flag (description, bit number, bit value)."""
        self.get_flags()
        return self.dflags

    @staticmethod
    def _as_list(attr):
        # attributes may be stored as a space-separated string (older products) or as a list
        if isinstance(attr, str):
            return attr.split(' ')
        return list(attr)

    def get_flags(self, ):
        """
        Build ``self.dflags``, a DataFrame indexed by flag name with columns
        ``description``, ``bit`` (bit number) and ``value`` (``1 << bit``).
        """

        pflags = self.product[self.flag_ID]
        names = self._as_list(pflags.attrs[self.names_])
        descriptions = self._as_list(pflags.attrs[self.description_])

        dflags = pd.DataFrame({'name': names})
        dflags['description'] = pd.Series(descriptions, dtype=object)
        dflags['bit'] = dflags.index
        dflags['value'] = [1 << int(bit) for bit in dflags['bit']]
        self.dflags = dflags.set_index('name')
        self.pflags = pflags

    @staticmethod
    def bitmask(mask, bitval, value):
        """
        Set (``value=True``) or clear (``value=False``) the bits ``bitval`` in ``mask``.

        :param mask: integer bitmask
        :param bitval: integer with the bits to modify set to 1
        :param value: boolean, set or clear
        :return: updated bitmask
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
        """
        Add a new flag to a bitmask DataArray.

        :param flags: bitmask xarray.DataArray (with 'flag_names' and 'flag_descriptions' attributes)
        :param boolean_cond: boolean xarray.DataArray, True where the flag is raised
        :param name: name of the new flag
        :param bitmask: bit number used to store the flag
        :param description: description of the flag
        :return: updated bitmask xarray.DataArray
        """
        attrs = dict(flags.attrs)
        flags = flags + (boolean_cond << bitmask)

        # copy the lists so that the original attributes are left untouched
        names = list(attrs.get('flag_names', []))
        descriptions = list(attrs.get('flag_descriptions', []))
        for list_ in (names, descriptions):
            list_.extend([''] * (bitmask + 1 - len(list_)))
        names[bitmask] = name
        descriptions[bitmask] = description

        attrs['flag_names'] = names
        attrs['flag_descriptions'] = descriptions
        flags.attrs = attrs
        return flags

    def compute_mask_value(self, **flags):
        """
        Compute the integer ``mask`` selecting the requested bits and the
        integer ``value`` expected for those bits.

        :param flags: flag_name=bool pairs
        :return: (mask, value) such that ``(bitmask & mask) == value`` selects the pixels
        """
        if not hasattr(self, 'dflags'):
            self.get_flags()

        mask = 0
        value = 0
        for flag_name, flag_ref in flags.items():
            bit_val = int(self.dflags.loc[flag_name, 'value'])
            mask = self.bitmask(mask, bit_val, True)
            value = self.bitmask(value, bit_val, flag_ref)
        self.mask = mask
        self.value = value
        return mask, value

    def get_mask(self, **flags):
        """
        Returns boolean xarray computed from ``**flags``, True where all the
        requested flag conditions are fulfilled.

        Example
        -------
        >>> masking_ = Masking(product)
        >>> mask_ = masking_.get_mask(ndwi=False, negative=False, nodata=True)

        :param flags: flag_name=bool pairs
        :return: boolean xarray.DataArray
        """

        mask, value = self.compute_mask_value(**flags)

        return (self.product[self.flag_ID] & mask) == value

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

        A pixel is masked if any bit of ``tomask`` is raised, or if none of the bits of ``tokeep`` is raised.

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

        flag_value_tomask = sum(1 << bitnum for bitnum in tomask)
        flag_value_tokeep = sum(1 << bitnum for bitnum in tokeep)

        if tomask and tokeep:
            mask = ((flags & flag_value_tomask) != 0) | ((flags & flag_value_tokeep) == 0)
        elif tokeep:
            mask = (flags & flag_value_tokeep) == 0
        elif tomask:
            mask = (flags & flag_value_tomask) != 0
        else:
            mask = xr.zeros_like(flags, dtype=bool)

        mask = mask.astype(_type)
        mask.attrs["long_name"] = "binary mask from flags"
        mask.attrs["description"] = "good pixels for mask == 0, bad pixels when mask == 1"
        mask.name = mask_name
        return mask
