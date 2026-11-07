# Products in an S3 bucket

Object storage and STAC catalogues are how data lakes and cloud platforms share satellite products. The
notebook below works the same way on a local machine, without moving the products:

- `grstbx-s3-bucket` (command installed with grstbx) serves the folders of products as a bucket with `rclone serve s3`, in place: **nothing is
  copied**. The server is read-only, listens on the local machine only and generates its own key, written
  to the AWS profile `grstbx-local` with the address of the server, so that the notebooks hold no secret;
- {func}`~grstbx.stac.build_catalog` writes a static STAC catalogue of the products, the collections
  `grs-l2a` and `grs-l2b`, with their URLs in the bucket. The L2B Items written by GRSl2bgen are reused,
  the other Items come from the attributes of the products, and each L2B Item is linked to its L2A Item;
- {func}`~grstbx.app.search_items` reads the catalogue from the bucket and selects the products by area,
  period, cloud cover, tile, version and collection;
- the products are opened from their URLs, lazily, and only the chunks needed are read.

The bucket `grstbx-s3` has one prefix per sensor and level, whatever the format of the products, and the
catalogue next to them:

```
s3://grstbx-s3/
    sentinel-2/l1c/    Level-1C images      -> /data/satellite/Sentinel-2/L1C
    sentinel-2/l2a/    GRS L2A products     -> /data/satellite/Sentinel-2/L2A_zarr
    sentinel-2/l2b/    GRSl2bgen L2B products -> /data/satellite/Sentinel-2/L2B_zarr
    stac/              catalog.json, grs-l2a/, grs-l2b/
```

Below each prefix, the tree of the batches is kept: `<tile>/YYYY/MM/DD/<name>/`, one folder per product with
its log and its STAC Item. Other sensors or levels get their own prefixes (e.g. `landsat-8/l2a/`):
give `grstbx-s3-bucket` the links as `<prefix>=<folder>` (`grstbx-s3-bucket --help` for the bucket name, the root
folder, the address and the profile). The command runs the shell script `s3_bucket.sh` shipped with grstbx,
whose path is given by `grstbx-s3-bucket --script`, to run it or adapt it without the command.

```bash
pip install "grstbx[remote]"
grstbx-s3-bucket       # leave it running; rclone must be installed
```

```python
import os
from grstbx.app import search_items
from grstbx.driver import open_zarr_image

os.environ['AWS_PROFILE'] = 'grstbx-local'
found = search_items('s3://grstbx-s3/stac/catalog.json', collections='grs-l2b',
                     bbox=(5.03, 43.40, 5.24, 43.56), start='2023-01-01', end='2023-12-31', max_cloud_cover=20)
ds = open_zarr_image(found.path[0])
```

The dashboard searches the same catalogue:

```bash
AWS_PROFILE=grstbx-local grstbx-viewer --stac s3://grstbx-s3/stac/catalog.json
```

The L2B products of Zarr format are read chunk by chunk. A NetCDF product in the bucket is read through
fsspec with h5netcdf, with many small requests: prefer Zarr for long time series.

A catalogue is a picture of the products when it was built: run {func}`~grstbx.stac.build_catalog` again
when products are added. It leaves out the incomplete products (L2A store without its required arrays, L2B
Zarr store without Item) and returns their list: a store that was not fully written can still open, the
missing chunks reading as empty data.

The notebook is rendered with the outputs of a run on the 1081 complete products (523 L2A, 558 L2B) of the
Gulf of Lion and the surroundings, read through `rclone serve s3` on the same machine. The products are not distributed with
grstbx: set the name of the bucket in the first cell to run it on your own data.

```{toctree}
:maxdepth: 1

s3_stac/grstbx_s3_stac
```

## Batches writing to the bucket

The batches of {doc}`batch` read their inputs from a bucket and write their products to it, with the same
URLs (`s3://grstbx-s3/sentinel-2/l2b`): each image is processed in a local work folder, and its product
folders are uploaded, the STAC Item last. To let them write to the local bucket, start it in writable mode:

```bash
grstbx-s3-bucket --write
```

The products are then written to the folders linked into the bucket. Once the batch is done, the
catalogue can be rebuilt from the Items read in the bucket, without access to the folders, and written
to the bucket directly:

```python
from grstbx import stac

stac.build_catalog(['s3://grstbx-s3/sentinel-2/l2a', 's3://grstbx-s3/sentinel-2/l2b'],
                   's3://grstbx-s3/stac')
```

Only the products with a STAC Item are seen in this way: the L2B products of GRSl2bgen (`--stac`, or
`stac=True` in the batches), and the L2A products of the batches run with `l2a_stac=True`
({func}`~grstbx.batch.write_l2a_item`). The other products are described from their attributes when the
catalogue is built from the local folders, as above.
