# Cropping and export

The functions of {mod}`grstbx.subset` extract an area of interest (AOI) from a GRS L2A product and
write it as a new, smaller L2A product, with the same layout and packing as the products of the
processor. The subsets can therefore be reopened with {class}`~grstbx.driver.L2grs` like any GRS
image.

```python
import grstbx

product, ancillary = grstbx.open_l2a(file)                    # lazy, with the ancillary data
aoi = grstbx.aoi_from_box(-2.25, 47.22, width=15000, height=12000)   # or gpd.read_file('aoi.geojson')

subset, subset_anc = grstbx.crop_l2a(product, aoi, ancillary)
subset = subset.load()                                        # reads the chunks within the AOI only

out = grstbx.export_l2a(subset, grstbx.subset_path(file, '/out', 'loire'), ancillary=subset_anc)
dc = grstbx.L2grs([out])                                      # reopen as any GRS product
```

## Crop

{func}`~grstbx.subset.crop_l2a` cuts the product to the bounding box of the AOI with
`rio.clip_box`, which slices the dask arrays: the result stays lazy and only the chunks intersecting
the AOI are read when it is loaded, whatever the size of the tile. With `mask_outside=True`
(default), the pixels of the bounding box outside the polygon(s) are flagged as missing, following
the GRS conventions:

- float variables (`Rrs`, `BRDFg`, angles...) set to NaN;
- bit 0 (`nodata`) of `flags` raised;
- `mask` set to 1 (invalid pixel).

The integer `flags` and `mask` keep their dtype. The ancillary data (coarse grid of about 10 km) are
cropped to the bounding box enlarged by one coarse cell, so that they can still be interpolated over
the whole subset.

## Export

{func}`~grstbx.subset.export_l2a` writes the subset with the exporters of GRS
(`grs.output.L2aProduct`; the [grs](https://github.com/Tristanovsk/grs) package is needed, it is
imported only when exporting). The format follows the extension of the output path:

| extension | output |
|---|---|
| `.nc` | NetCDF: `<name>/<name>.nc` (main) and `<name>/<name>_anc.nc` (ancillary), zlib compression |
| `.zarr` | Zarr: `<name>.zarr` with the full resolution in group `0`, the overviews (pyramid) and the group `ancillary` |

`Rrs`, `BRDFg`, `aot550`, the angles and `dem` are packed as `int16` with `scale_factor` /
`add_offset` (1e-5 sr{sup}`-1` for `Rrs`): the export of a subset of a GRS product is lossless.
{func}`~grstbx.subset.subset_path` builds the output path from the name of the input product and
the name of the AOI:

```python
grstbx.subset_path(file, '/out', 'loire')              # /out/loire/<product>_loire/<product>_loire.nc
grstbx.subset_path(file, '/out', 'loire', fmt='zarr')  # /out/loire/<product>_loire.zarr
```

{func}`~grstbx.subset.export_rrs_geotiff` writes $R_{rs}$ as a multi-band float32 GeoTIFF (one band
per wavelength, named `Rrs_<wl>`) for GIS software; `water_only=True` keeps the valid water pixels
only.

## Series of images

{func}`~grstbx.subset.crop_and_export` opens, crops and exports a product in one call; existing
outputs are skipped unless `overwrite=True`:

```python
import glob

for file in sorted(glob.glob('/data/satellite/Sentinel-2/L2A/30TWT/2024/*/*/*.zarr')):
    grstbx.crop_and_export(file, aoi, grstbx.subset_path(file, '/out', 'loire', fmt='zarr'))
```

## Notebook

The notebook below runs the whole workflow on a Sentinel-2 tile: preview at a coarse pyramid level,
drawing of the AOI on an interactive map, crop, export, check of the exported product and batch
processing. It is rendered with the outputs of its last run, where no polygon was drawn (the default
box is used; the interactive widgets are not functional in this page). The input image is not
distributed with grstbx: set the paths of the settings cell to run it on your own data. The notebook
is [`notebook/grstbx_l2a_visu_subset_export.ipynb`](https://github.com/Tristanovsk/grstbx/blob/main/notebook/grstbx_l2a_visu_subset_export.ipynb).

```{toctree}
:maxdepth: 1

subset_export/grstbx_l2a_visu_subset_export
```
