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
- the batch itself: images already processed, too cloudy or without CAMS file are skipped, an existing
  L2A product is reused, a failing step is logged and does not stop the batch, and a summary table is
  saved;
- a **profiling file** per image (`<l2b_dir>/profiles/<tile>/<image>_profile.json`): wall time, CPU time
  and peak memory of the L2A and L2B steps, and the memory curve sampled during the processing, to size
  the machine or the number of images processed in parallel;
- a quick look at the L2A and L2B products of one image;
- the same batch from the command line, one `grs` and one `GRSl2bgen` process per image, profiled with
  GNU `time -v`, which is more robust for long batches.

```python
import grs
import GRSl2bgen

for date, row in selection.iterrows():                 # one row per Level-1 image
    profiler = Profiler(row['name'])
    with profiler.step('L2A'):
        process = grs.Process()
        process.execute(row['path'], odir=row['l2a'], cams_file=row['cams_file'], resolution=20)
        process.write_output()                         # L2A Zarr store <l2a_folder>/<name>.zarr
    with profiler.step('L2B'):
        GRSl2bgen.Process(l2a_folder, row['l2b']).run()   # L2B NetCDF <l2b_folder>/<name>.nc
    profiler.write(row['profile'])                     # wall and CPU times, peak memory per step
```

The notebook is rendered with the outputs of a short run: the three images of 1 and 2 August 2025 of tile
31TEJ (Gulf of Lion), full tiles at 20 m with GRS 3.1.0 and GRSl2bgen 1.2 on a 40-core machine. The L2A
step took 2 to 4 minutes per image with a memory peak of 10 to 12 GB, and the L2B step about 70 s with a
peak of 7.7 GB. The input images are not distributed with grstbx: set the paths of the configuration cell
to run it on your own data.

```{toctree}
:maxdepth: 1

batch/grs_l1c_l2a_l2b_batch
```
