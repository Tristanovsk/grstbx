# Selecting products

{class}`~grstbx.datalake.SelectFiles` lists the GRS products stored on disk and filters them by date
and cloud coverage. The file names are parsed into a {class}`pandas.DataFrame` indexed by date
(see {ref}`file naming <file-naming>`).

## Files of a folder

```python
import grstbx

select = grstbx.SelectFiles('/data/satellite/Sentinel-2/L2A')
select.list_folder(pattern='*.nc')
select.file_list            # satellite, level, tile, cloud_coverage, version, abspath
```

## Files of a tile

`list_tile` expects the data lake layout `root/product/tile/YYYY/MM/DD/<files>`:

```python
select = grstbx.SelectFiles('/datalake/watcal')
select.list_tile(product='S2-L2GRS', tile='31TEJ', pattern='*.nc')
```

## Filtering by date and cloud coverage

```python
files = select.list_file_path(('2022-07-01', '2022-08-31'), cc_max=50)
```

`list_file_path` filters `select.file_list` in place and returns the absolute paths of the
selected files, ready to be passed to {class}`~grstbx.driver.L2grs`.

In a notebook, `select.select_dates()` returns a panel `DatetimeRangePicker` spanning the listed
dates, to choose the period interactively:

```python
picker = select.select_dates()
picker                                  # display the widget
files = select.list_file_path(picker.value)
```
