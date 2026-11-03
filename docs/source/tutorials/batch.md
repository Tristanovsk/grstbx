# Batch processing

The notebook below produces the products explored in the other tutorials: it processes a time series of
Sentinel-2 L1C images (or Landsat Collection 2 Level-1 images) of one or several tiles, image by image,
into a GRS **L2A** product with [GRS](https://grs.readthedocs.io/en/latest/), then into a **L2B** product
with [GRSl2bgen](https://grsl2bgen.readthedocs.io/en/latest/) as soon as the L2A product is written. It
merges the batch notebooks of the two processors:

- configuration of the batch: folders, tiles, date range, CAMS and DEM files, options of the two
  processors, and check of the look-up tables of GRS;
- list of the Level-1 images found on disk (`<idir>/<tile>/YYYY/MM/DD/<image>`) with their cloud cover,
  and selection of a date range;
- names of the L2A and L2B products, as with the `grs` and `GRSl2bgen` command lines, one folder per
  product (`<tile>/YYYY/MM/DD/<name>/`) with its log and its STAC Item (written by GRSl2bgen for the L2B,
  by grstbx for the L2A with `l2a_stac=True`);
- inputs and outputs on local disks or in an S3 bucket (see below);
- the state of the products of each image (`l2a_done`, `l2b_done`), which decides the steps to run: GRS
  for the images without complete L2A product, GRSl2bgen for those without complete L2B product, an
  existing L2A product being reused (`overwrite_l2b` recomputes the L2B products alone, e.g. with a new
  version of GRSl2bgen);
- the batch itself: too cloudy images or images without CAMS file are skipped when their L2A product is
  to be computed, a failing step is logged and does not stop the batch, and a summary table is saved;
- a **profiling file** per image (`<l2b_dir>/profiles/<tile>/<image>_profile.json`): wall time, CPU time
  and peak memory of the L2A and L2B steps, and the memory curve sampled during the processing, to size
  the machine or the number of images processed in parallel;
- a quick look at the L2A and L2B products of one image;
- the same batch from the command line, one `grs` and one `GRSl2bgen` process per image, profiled with
  GNU `time -v`, which is more robust for long batches.

```python
from grstbx import batch

images = batch.list_images('/data/L1C', ['31TEJ']).loc['2025-08-01':'2025-08-04']
run = batch.Batch('/data/L2A', '/data/L2B', cams_file_template='/data/cams/cams_forecast_{date:%Y-%m}.nc')
plan = run.plan_level1(images)    # names, state of the products (l2a_done, l2b_done), steps to run
summary = run.run(plan)           # GRS then GRSl2bgen, image by image, a profile per image
```

The notebooks of this section use the module {mod}`grstbx.batch`: listing of the Level-1 images
({func}`~grstbx.batch.list_images`) or of the L2A products ({func}`~grstbx.batch.list_l2a_products`),
names of the products as with the command lines of GRS and GRSl2bgen, completeness checks
({func}`~grstbx.batch.l2a_complete`, {func}`~grstbx.batch.l2b_complete`), processing of one image
({func}`~grstbx.batch.run_grs`, {func}`~grstbx.batch.run_grsl2bgen`), profiles
({class}`~grstbx.batch.Profiler`, {func}`~grstbx.batch.read_profiles`, {func}`~grstbx.batch.plot_memory`)
and the batch itself ({class}`~grstbx.batch.Batch`, also the shell commands of a plan). `grs` and
`GRSl2bgen` are imported only to process the images.

The notebook is rendered with the outputs of a short run: the three images of 1 and 2 August 2025 of tile
31TEJ (Gulf of Lion), full tiles at 20 m with GRS 3.1.0 and GRSl2bgen 1.2 on a 40-core machine. The L2A
step took 2 to 4 minutes per image with a memory peak of 10 to 12 GB, and the L2B step about 70 s with a
peak of 7.7 GB. The input images are not distributed with grstbx: set the paths of the configuration cell
to run it on your own data.

```{toctree}
:maxdepth: 1

batch/grs_l1c_l2a_l2b_batch
```

## Inputs and outputs in an S3 bucket

The roots of the inputs (`idir`, `rootdir`) and of the outputs (`l2a_dir`, `l2b_dir`, `odir`) of the two
batch notebooks of {mod}`grstbx.batch` can be bucket URLs instead of local folders, e.g.
`s3://grstbx-sentinel2/L1C` and `s3://grstbx-sentinel2/L2B_zarr`. The two can be mixed, e.g. local Level-1
images and products written to a bucket. Paths are read and written through fsspec
(`pip install "grstbx[remote]"`). The credentials and the endpoint come from the environment (e.g.
`AWS_PROFILE`, see {doc}`s3_stac`) or from `storage_options`. GRS and GRSl2bgen work on local files, so
{class}`~grstbx.batch.Batch` stages the images one at a time:

1. the Level-1 image, or the L2A product when only its L2B product is computed, is downloaded to
   `work_dir`, a local folder (a temporary folder by default: choose a disk with a few GB free per image);
2. GRS and GRSl2bgen process it there, as for local folders;
3. each product folder is uploaded to the bucket: the chunks first, then the Zarr metadata, then the
   STAC Item last. A product interrupted during the upload is therefore not complete
   ({func}`~grstbx.batch.l2a_complete`, {func}`~grstbx.batch.l2b_complete`), and it is processed again
   at the next run;
4. the local copies are removed, and the profile of the image and the summary of the batch are written
   to the bucket.

```python
import os
from grstbx import batch

os.environ['AWS_PROFILE'] = 'grstbx-local'
images = batch.list_images('s3://grstbx-sentinel2/L1C', ['31TEJ']).loc['2025-08-01':'2025-08-04']
run = batch.Batch('s3://grstbx-sentinel2/L2A_zarr', 's3://grstbx-sentinel2/L2B_zarr',
                  cams_file_template='/data/cams/cams_forecast_{date:%Y-%m}.nc',
                  work_dir='/scratch/grstbx', l2a_stac=True)
summary = run.run(run.plan_level1(images))
```

The local bucket of {doc}`s3_stac` serves the folders of products in place. Start it with `WRITE=1 bash
s3_bucket.sh` to let the batches write to it: the products are then written to the local folders linked
into the bucket. Two limits: the check of the products in a bucket costs about 20 requests per L2A product
(about 0.15 s per product with the local bucket), and the shell commands
({meth}`~grstbx.batch.Batch.commands`) need local folders. To run them on a bucket, mount it as a folder
(`rclone mount`).

### STAC catalogue of the products

GRSl2bgen writes the STAC Item of each L2B product (`stac=True`). With `l2a_stac=True`, the batch also
writes the Item of each L2A product that it computes ({func}`~grstbx.batch.write_l2a_item`, same content
as the Items of {func}`~grstbx.stac.product_item`), next to the product: `<name>/<name>.json`. Each L2B
Item has a `derived_from` link to the URL of its L2A product in the bucket. Once a batch is done, the
static catalogue of {doc}`s3_stac` is rebuilt from the Items alone, read from the bucket, without
downloading any product, and written to the bucket:

```python
from grstbx import stac
from grstbx.app import search_items

stac.build_catalog(['s3://grstbx-sentinel2/L2A_zarr', 's3://grstbx-sentinel2/L2B_zarr'],
                   's3://grstbx-sentinel2/stac')
found = search_items('s3://grstbx-sentinel2/stac/catalog.json', collections='grs-l2b', tile='31TEJ')
```

Products without an Item are not seen by a catalogue built from a bucket. These are, for example, L2A
products computed without `l2a_stac=True`, or with the command line of GRS. Build the catalogue from the
local folders instead (`{'/data/L2A_zarr': 's3://grstbx-sentinel2/L2A_zarr'}`), which describes them from
their attributes.

## L2A to L2B: notebook of GRSl2bgen

When the L2A products already exist, e.g. produced by the scripts below or to recompute the L2B products
with a new version of GRSl2bgen, the batch notebook of the
[GRSl2bgen documentation](https://grsl2bgen.readthedocs.io/en/latest/tutorials/grsl2bgen_batch_processing.html)
processes them alone into L2B products, for one or several tiles and a date range. It reads the products of
both major versions of GRS, detected product by product: the Zarr stores of GRS 3.x and the NetCDF files of
GRS 2.x (`<name>.nc` and `<name>_anc.nc`). Incomplete products are left out (e.g. a Zarr store whose
levels lack the `Rrs` and `mask` arrays after a failed GRS run), one GRS version is kept per image, the
products already processed (STAC Item present) are skipped, and its last section gives the same batch from
the command line, one `GRSl2bgen` process per product.

```{toctree}
:maxdepth: 1

batch/grsl2bgen_batch
```

## L1C to L2A: notebooks and bash scripts for GRS 2.x and 3.x

The batch of GRS alone, from L1C to L2A, is given for each major version of GRS, as a notebook (the batch
notebook of the GRS documentation) and as a bash script. For long batches, the script is more robust than
a notebook: each image runs in its own `grs` process, so that the memory is fully released between
images, the batch runs without a Jupyter kernel, and several images can be processed in parallel.

| | GRS 2.x | GRS 3.x |
|---|---|---|
| L2A product | NetCDF `<name>.nc` and ancillary `<name>_anc.nc` | Zarr store `<name>.zarr` (pyramid) |
| LUT folder | `config.yml` of the installed package | `~/.config/grs/config.yml` (`grs_config`) or `GRS_DATA_ROOT` |
| option of the aerosol grid | none | `--aot_resolution` |
| memory, full tile at 20 m | 30 to 32 GB | 9 to 13 GB |

The scripts skip the complete products, too cloudy images and images without CAMS file, check each
product after the run (`grs` exits with 0 even when it fails), write a summary CSV file, and profile each
image with GNU `time -v` (wall time, CPU times, maximum resident memory).

```{toctree}
:maxdepth: 1
:caption: GRS 2.x

batch/grs_s2_batch_v2
batch/grs_batch_v2
```

```{toctree}
:maxdepth: 1
:caption: GRS 3.x

batch/grs_s2_batch_v3
batch/grs_batch_v3
```
