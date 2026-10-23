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
  product (`<tile>/YYYY/MM/DD/<name>/`) with its log and, for the L2B, its STAC Item;
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
