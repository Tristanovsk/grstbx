# Export

The notebook below exports the images of an area of interest and a date range to a shared folder
(Google Drive), for people who do not have access to the storage of the products:

- area: point and box size, bounding box or vector file;
- date range and data levels: `l1c`, `l2a`, `l2b`;
- tiles: every tile intersecting the area (images exported tile by tile, without mosaic), or the tile
  covering it best; footprints read from the STAC Items, the products or the L1C metadata and cached;
- images: one per tile and date, latest GRS version unless a version is set, cloud-cover threshold;
- exports cut to the area:
  - L1C: zip of GeoTIFFs (one per resolution) with the metadata files `MTD_MSIL1C.xml` and `MTD_TL.xml`;
  - L2A: written with the GRS writers ({func}`~grstbx.subset.crop_l2a`, {func}`~grstbx.subset.export_l2a`),
    NetCDF by default;
  - L2B: NetCDF keeping the `int16` packing of the product;
  - zipped Zarr stores (`.zarr.zip`, read with `xr.open_zarr(zarr.storage.ZipStore(path, mode='r'))`)
    as an option for L2A and L2B: unzipped Zarr stores (thousands of small files) are avoided;
- manifest of the exports (`manifest_<start>_<end>.csv`) and upload with `rclone copy`; existing exports
  are skipped, so an interrupted run can be resumed.

```python
# settings cell
site = 'loire_estuary'
lon, lat = -2.25, 47.22          # centre of the box (deg)
width, height = 15000, 12000     # m
bbox = None                      # e.g. (-2.4, 47.15, -2.0, 47.32): replaces the box above
aoi_file = None                  # e.g. 'aoi.geojson': replaces the box above
start_date, end_date = '2017-01-01', '2017-01-20'
levels = ['l1c', 'l2a', 'l2b']
tile_choice = 'all'              # 'all' (every tile intersecting the area) or 'best'
fmt = 'nc'                       # 'nc' or 'zarr.zip' (L2A and L2B)
drive_remote, drive_folder = 'mag_watqual:', 'grstbx_exports'
```

The notebook reads the local `grstbx-s3` bucket (see {doc}`s3_stac`), with the Loire estuary (tile 30TWT) as
example. It is rendered without outputs; rclone needs a Google Drive remote (`rclone config`).

The same export from the TREX datalake of CNES, run on the compute nodes as a SLURM job array with the upload
from a login node, is in the HPC section: {doc}`../hpc/cnes_trex/grstbx_trex_export_gdrive`.

```{toctree}
:maxdepth: 1

export/grstbx_s3_export_gdrive
```
