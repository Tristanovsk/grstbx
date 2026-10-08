# Masking with the bitmask flags

GRS stores the pixel classification as a single integer `flags` raster in which bit `i` is set when
condition `flag_names[i]` holds. {class}`~grstbx.masking.Masking` decodes those bits into boolean
masks.

## Available flags

```python
import grstbx

masking_ = grstbx.Masking(product)          # product: xarray.Dataset with a 'flags' variable
masking_.print_info()
```

`print_info` returns a {class}`pandas.DataFrame` indexed by flag name, with the description, the
bit number and the bit value (`1 << bit`) of each flag.

## Boolean masks from flag names

{meth}`~grstbx.masking.Masking.get_mask` takes `flag_name=bool` pairs and returns a boolean
DataArray, `True` where **all** the requested conditions are fulfilled:

```python
# pixels that are not flagged as land (ndwi) nor as high clouds
water = masking_.get_mask(ndwi=False, hicld=False)
Rrs = product.Rrs.where(water)
```

Internally, the requested bits are combined into an integer `mask` and the expected `value`
({meth}`~grstbx.masking.Masking.compute_mask_value`), and the pixels are selected with
`(flags & mask) == value`.

## Binary masks from bit numbers

{meth}`~grstbx.masking.Masking.create_mask` builds a `uint8` mask from bit numbers, with the
convention **0 for good pixels, 1 for bad pixels**. A pixel is masked if any bit of `tomask` is
raised, or if none of the bits of `tokeep` is raised:

```python
mask = grstbx.Masking.create_mask(product.flags,
                                  tomask=[0, 2, 11],
                                  tokeep=[3],
                                  mask_name='mask_from_flags')
Rrs = product.Rrs.where(mask == 0)
```

## Adding a flag

User-defined conditions can be stored in a free bit of the bitmask with
{meth}`~grstbx.masking.Masking.add_flag`; the `flag_names` and `flag_descriptions` attributes are
updated accordingly:

```python
turbid = product.Rrs.sel(wl=865, method='nearest') > 0.01
product['flags'] = grstbx.Masking.add_flag(product.flags, turbid, name='turbid', bitmask=20,
                                           description='Rrs(865) > 0.01 sr-1')
grstbx.Masking(product).get_mask(turbid=True)
```
