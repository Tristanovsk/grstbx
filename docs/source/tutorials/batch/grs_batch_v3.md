# Bash batch with GRS 3.x

This script is the command-line version of the batch notebook of GRS
([Batch processing of Sentinel-2 images](https://grs.readthedocs.io/en/v3.x/examples/batch.html)) for
**GRS 3.x**: it processes a time series of Sentinel-2 L1C images of one or several tiles into GRS L2A
products, with one `grs` process per image. Compared with a notebook, the memory is fully released after
each image, the batch runs without a Jupyter kernel (e.g. with `nohup` or in a `tmux` session), and
several images can be processed in parallel.

## Before running it

- GRS 3.x installed, the `grs` command in the `PATH` (`grs --version` prints `grs_3.x.y`);
- the look-up tables (LUT) of GRS found: GRS 3 reads their folder from the user configuration
  `~/.config/grs/config.yml` or from the environment variable `GRS_DATA_ROOT`. Create the configuration
  and check the LUTs with

  ```bash
  grs_config --init /data/grs/grsdata     # folder of the LUTs
  grs_config --download                   # download the missing LUTs from Zenodo
  grs_config                              # configuration in use and LUT check
  ```

- the CAMS file of the month of each image (see the
  [CAMS tutorial of GRS](https://grs.readthedocs.io/en/v3.x/examples/auxdata.html)), and optionally a
  DEM GeoTIFF of each tile;
- GNU `time` (`/usr/bin/time`, package `time` of the Linux distributions) for the profiles.

## Configuration

The first block of the script sets the batch:

| Variable | Description |
|---|---|
| `idir`, `odir` | root folders of the L1C images (`<idir>/<tile>/YYYY/MM/DD/<image>.SAFE`) and of the L2A products |
| `tiles` | tiles to process, separated by spaces |
| `start`, `end` | date range (inclusive), from the `YYYY/MM/DD` folders |
| `cams_template` | CAMS file of the month, with the date format of `date` (`%Y-%m`) |
| `dem_template` | DEM of the tile, `TILE` replaced by the tile; a missing file means no DEM (fine over the sea) |
| `resolution` | pixel size (m) of the L2A product; to be given, the default of `grs` is 60 m |
| `aot_resolution` | pixel size (m) of the coarse grid of the aerosol retrieval, multiple of `resolution` |
| `max_cloud_cover` | skip the images whose L1C cloud cover (0-1) is greater or equal |
| `scale_aot`, `levname`, `suffix` | options of the same name of `grs` (`suffix=""`: `_V<grs version>`) |
| `jobs` | number of images processed in parallel |

## What it does

For each L1C image of the tiles and date range, `process_one`:

1. builds the name of the L2A product as `grs` does (`MSIL1C` replaced by `MSIL2AGRS`, plus the
   version), and its folder `<odir>/<tile>/YYYY/MM/DD/<name>/`, which holds the Zarr store
   `<name>.zarr` and the log `log_file.log` written by `grs`;
2. **skips** the image if its product is complete (root metadata, and the `Rrs` and `mask` arrays of
   every level of the pyramid), if its CAMS file is missing, or if its cloud cover (from
   `MTD_MSIL1C.xml`) reaches `max_cloud_cover`;
3. runs `grs` through GNU `time -v` from the log folder of the image, `<odir>/logs/<tile>/<image>/`:
   `grs` writes its `error.log` to the current folder, which would be overwritten by the other images;
4. checks the product: `grs` exits with 0 even when the processing fails (the error is only logged), so
   the image is **done** when its product is complete, and **failed** otherwise; a partial product is
   removed so that the image is retried at the next run;
5. appends a line to the summary CSV file `<odir>/grs_batch_<date>.csv`: date, tile, image, status, wall
   time and maximum resident memory (kB) of the run, and product.

The images are passed to `process_one` by `xargs -P $jobs`. A full tile at 20 m needs 9 to 13 GB of
memory with GRS 3.1 (see the run below): set `jobs` so that `jobs` × 13 GB stays within the free memory.

```{literalinclude} grs_batch_v3.sh
:language: bash
```

{download}`Download the script <grs_batch_v3.sh>`

## Run it

```bash
bash grs_batch_v3.sh                                   # in the foreground
nohup bash grs_batch_v3.sh > grs_batch.log 2>&1 &      # in the background, after logging out
```

The batch can be stopped at any time (`Ctrl+C`, or `kill` of the `xargs` process) and restarted: the
complete products are skipped, and an image stopped while being processed starts again from scratch.

## Outputs and profiles

```
<odir>/<tile>/YYYY/MM/DD/<name>/
    <name>.zarr/              L2A product (pyramid of levels 0, 1, 2... and ancillary group)
    log_file.log              log of grs
<odir>/logs/<tile>/<image>/
    grs.out                   console output of grs
    error.log                 errors of grs
    profile.txt               GNU time report: wall time, CPU times, maximum resident memory...
<odir>/grs_batch_<date>.csv   summary of the batch
```

The `profile.txt` files give the resources of each image, e.g. to choose `jobs`:

```bash
grep -H -E 'Elapsed|Maximum resident' <odir>/logs/*/*/profile.txt
```

Run of the script on the three images of 1 and 2 August 2025 of tile 31TEJ, two at a time (`jobs=2`),
with GRS 3.1.0 on a 40-core machine:

```text
$ bash grs_batch_v3.sh
15:56:06 done S2B_MSIL1C_20250801T102559_N0511_R108_T31TEJ_20250801T140637.SAFE (4:11.79, 9128 MB)
15:57:53 done S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE (5:59.56, 13051 MB)
15:58:44 done S2C_MSIL1C_20250802T105051_N0511_R051_T31TEJ_20250802T142718.SAFE (2:38.02, 9194 MB)
summary: /data/satellite/Sentinel-2/L2A_zarr/grs_batch_20261023T155154.csv
      3 done
$ bash grs_batch_v3.sh          # second run: all done
15:58:44 skipped S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE 
15:58:44 skipped S2B_MSIL1C_20250801T102559_N0511_R108_T31TEJ_20250801T140637.SAFE 
15:58:44 skipped S2C_MSIL1C_20250802T105051_N0511_R051_T31TEJ_20250802T142718.SAFE 
summary: /data/satellite/Sentinel-2/L2A_zarr/grs_batch_20261023T155844.csv
      3 skipped
```
