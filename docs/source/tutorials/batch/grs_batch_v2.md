# Bash batch with GRS 2.x

This script is the command-line version of the batch notebook of GRS
([Batch processing of Sentinel-2 images](https://grs.readthedocs.io/en/latest/examples/batch.html)) for
**GRS 2.x**: it processes a time series of Sentinel-2 L1C images of one or several tiles into GRS L2A
products, with one `grs` process per image. Compared with a notebook, the memory is fully released after
each image, the batch runs without a Jupyter kernel (e.g. with `nohup` or in a `tmux` session), and
several images can be processed in parallel. For GRS 3.x, see {doc}`grs_batch_v3`.

## Before running it

- GRS 2.x installed, the `grs` command in the `PATH` (`grs --version` prints `grs_2.x.y`);
- the look-up tables (LUT) of GRS found: GRS 2 reads their folder (`grsdata`) and file names
  (`toa_lut`, `trans_lut`) from the `config.yml` file of the installed package. Find it and check the
  LUTs with

  ```bash
  config=$(python -c "from importlib.resources import files; print(files('grs') / 'config.yml')")
  grep -A3 '^path:' "$config"            # edit grsdata if the LUTs are elsewhere
  ```

- the CAMS file of the month of each image (see the
  [CAMS tutorial of GRS](https://grs.readthedocs.io/en/latest/examples/auxdata.html)), and optionally a
  DEM GeoTIFF of each tile;
- GNU `time` (`/usr/bin/time`, package `time` of the Linux distributions) for the profiles, and
  optionally `ncdump` (netCDF utilities) to check that the products can be read.

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
| `max_cloud_cover` | skip the images whose L1C cloud cover (0-1) is greater or equal |
| `scale_aot`, `levname`, `suffix` | options of the same name of `grs` (`suffix=""`: `_V<grs version>`) |
| `jobs` | number of images processed in parallel |

## What it does

For each L1C image of the tiles and date range, `process_one`:

1. builds the name of the L2A product as `grs` does (`MSIL1C` replaced by `MSIL2AGRS`, plus the
   version), and its folder `<odir>/<tile>/YYYY/MM/DD/<name>/`, which holds the NetCDF product
   `<name>.nc`, its ancillary data (CAMS) `<name>_anc.nc` and the log `log_file.log` written by `grs`;
2. **skips** the image if its product is complete (both files present and, with `ncdump`, readable with
   the `Rrs` and `mask` variables), if its CAMS file is missing, or if its cloud cover (from
   `MTD_MSIL1C.xml`) reaches `max_cloud_cover`;
3. runs `grs` through GNU `time -v` from the log folder of the image, `<odir>/logs/<tile>/<image>/`:
   `grs` writes its `error.log` to the current folder, which would be overwritten by the other images;
4. checks the product: `grs` exits with 0 even when the processing fails (the error is only logged), so
   the image is **done** when its product is complete and its `error.log` reports no fatal error, and
   **failed** otherwise; a partial product is removed so that the image is retried at the next run;
5. appends a line to the summary CSV file `<odir>/grs_batch_<date>.csv`: date, tile, image, status, wall
   time and maximum resident memory (kB) of the run, and product.

The images are passed to `process_one` by `xargs -P $jobs`. A full tile at 20 m needs 30 to 32 GB of
memory with GRS 2.2 (see the run below): set `jobs` so that `jobs` × 32 GB stays within the free memory. The script sets
`MALLOC_ARENA_MAX=2`, which lets the memory freed by the processing threads be reused (GRS 3 sets it
itself).

```{literalinclude} grs_batch_v2.sh
:language: bash
```

{download}`Download the script <grs_batch_v2.sh>`

## Run it

```bash
bash grs_batch_v2.sh                                   # in the foreground
nohup bash grs_batch_v2.sh > grs_batch.log 2>&1 &      # in the background, after logging out
```

The batch can be stopped at any time (`Ctrl+C`, or `kill` of the `xargs` process) and restarted: the
complete products are skipped, and an image stopped while being processed starts again from scratch.

## Outputs and profiles

```
<odir>/<tile>/YYYY/MM/DD/<name>/
    <name>.nc                 L2A product
    <name>_anc.nc             ancillary data (CAMS)
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

Run of the script on the three images of 1 and 2 August 2025 of tile 31TEJ, one at a time (`jobs=1`),
with GRS 2.2.0 on a 40-core machine:

```text
$ bash grs_batch_v2.sh
16:09:38 done S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE (10:48.73, 30014 MB)
16:17:24 done S2B_MSIL1C_20250801T102559_N0511_R108_T31TEJ_20250801T140637.SAFE (7:45.54, 32023 MB)
16:24:24 done S2C_MSIL1C_20250802T105051_N0511_R051_T31TEJ_20250802T142718.SAFE (6:59.67, 32437 MB)
summary: /data/satellite/Sentinel-2/L2A_nc/grs_batch_20261023T155849.csv
      3 done
$ bash grs_batch_v2.sh          # second run: all done
16:24:37 skipped S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE 
16:24:38 skipped S2B_MSIL1C_20250801T102559_N0511_R108_T31TEJ_20250801T140637.SAFE 
16:24:38 skipped S2C_MSIL1C_20250802T105051_N0511_R051_T31TEJ_20250802T142718.SAFE 
summary: /data/satellite/Sentinel-2/L2A_nc/grs_batch_20261023T162434.csv
      3 skipped
```
