# grstbx

[![PyPI](https://img.shields.io/pypi/v/grstbx)](https://pypi.org/project/grstbx/)
[![Python](https://img.shields.io/pypi/pyversions/grstbx)](https://pypi.org/project/grstbx/)
[![Documentation](https://readthedocs.org/projects/grstbx/badge/?version=latest)](https://grstbx.readthedocs.io)
[![License: GPL v3](https://img.shields.io/github/license/Tristanovsk/grstbx)](https://github.com/Tristanovsk/grstbx/blob/main/LICENSE)

**grstbx** gathers the scientific code used to visualize and post-process the images produced by the
[GRS processor](https://github.com/Tristanovsk/grs) (Glint Removal for Sentinel-2-like sensors):

- **L2A** products: remote-sensing reflectance R<sub>rs</sub>(λ) corrected for the atmosphere and the
  sunglint (GRS multispectral, hGRS hyperspectral for EnMAP and PRISMA),
- **L2B** products: water-quality parameters (chlorophyll-a, suspended matter, ...) retrieved from
  R<sub>rs</sub>.

It provides:

- the selection of the products stored in a data lake by tile, date and cloud coverage,
- the loading of image series as lazy, dask-backed multi-temporal xarray datacubes, from NetCDF files
  or single-resolution / multiscale (pyramid) Zarr stores,
- the decoding of the GRS bitmask `flags` into boolean masks,
- the cropping of L2A products to an area of interest and the export of the subsets as GRS products
  (NetCDF, Zarr) or GeoTIFF,
- interactive viewers for Jupyter notebooks, based on holoviews, datashader and panel, to browse the
  images by date and band and to draw areas or points of interest, and a web dashboard to explore L2A
  and L2B products and time series without notebook (`grstbx-viewer`),
- matchups between satellite and in situ data, and miscellaneous helpers: bounding boxes, AERONET-OC
  readers, solar irradiance, DEM-based illumination, plotting.

![grstbx viewer](https://raw.githubusercontent.com/Tristanovsk/grstbx/main/illustration/grstbx_visual_tool.gif)

## Installation

grstbx requires Python >= 3.11. Install it from [PyPI](https://pypi.org/project/grstbx/):

```
pip install grstbx                 # library, viewers and web dashboard (grstbx-viewer)
pip install "grstbx[notebook]"     # + JupyterLab environment to use the viewers in notebooks
```

The geospatial libraries (GDAL, PROJ, GEOS) are compiled libraries: in a conda environment, install
them from conda-forge first, then grstbx with pip:

```
conda create -n grstbx -c conda-forge python=3.12 gdal rasterio pyproj cartopy netcdf4
conda activate grstbx
pip install "grstbx[notebook]"
```

To run the notebooks of the repository or to develop grstbx, clone it and create the conda environment
of `environment.yml`, which installs all the dependencies (including Jupyter and xesmf) and grstbx in
editable mode:

```
git clone https://github.com/Tristanovsk/grstbx.git
cd grstbx
conda env create -f environment.yml
conda activate grstbx
```

| extra | content | usage |
|---|---|---|
| `notebook` | JupyterLab, ipykernel, jupyter_bokeh | run the notebooks and the viewers |
| `regrid` | xesmf | `grstbx.utils.Reproj.regridding`; xesmf needs ESMF: `conda install -c conda-forge xesmf` |
| `docs` | Sphinx, sphinx-book-theme, myst-nb... | build the documentation |
| `all` | `notebook` and `docs` | |

The export of subsets (`grstbx.export_l2a`, `grstbx.crop_and_export`) uses the writers of
[GRS](https://github.com/Tristanovsk/grs): it needs `grs >= 3.0.1` in the same environment. To use the
environment in Jupyter, register it as a kernel: `python -m ipykernel install --user --name=grstbx`.

See the [installation page](https://grstbx.readthedocs.io/en/latest/installation.html) for the details
and troubleshooting.

## Quick start

```python
import grstbx

dc = grstbx.L2grs(files)                    # list of GRS L2A products
dc.get_l2a_datacube(subset=aoi)             # aoi: geopandas.GeoDataFrame
water = grstbx.Masking(dc.datacube).get_mask(ndwi=False)
Rrs = dc.datacube.Rrs.where(water)

viewer = grstbx.visual.ViewSpectral(Rrs, reproject=True)   # interactive viewer, in Jupyter
viewer.visu()
```

## Web dashboard

`grstbx-viewer` explores GRS L2A and L2B products (Zarr or NetCDF, hGRS NetCDF), single images, time
series or datacubes, in the web browser, without notebook:

```
grstbx-viewer /path/to/product.zarr              # or: grstbx-viewer --root /data/satellite
grstbx-viewer "/data/L2B/BEFR/2024/*/*/*.nc"     # time series of L2B products
grstbx-viewer /data/L2A_zarr --tile 31TEJ        # time series of a tile, folders <tile>/YYYY/MM/DD/<product>
```

Pan and zoom the images over a basemap, hover them to plot the spectrum (L2A) or the time series (L2B)
under the mouse, change the RGB bands, contrast and brightness, draw points and polygons, and download
their values and statistics. See the
[dashboard tutorial](https://grstbx.readthedocs.io/en/latest/tutorials/dashboard.html).

![grstbx-viewer](https://raw.githubusercontent.com/Tristanovsk/grstbx/main/illustration/grstbx_viewer_dashboard.png)

## Documentation

https://grstbx.readthedocs.io: [installation](https://grstbx.readthedocs.io/en/latest/installation.html),
[GRS products](https://grstbx.readthedocs.io/en/latest/products.html), tutorials (L2A image, web
dashboard, L2B datacube, cropping and export, validation with in situ data, case studies, hGRS
products), [API reference](https://grstbx.readthedocs.io/en/latest/api.html) and
[history](https://grstbx.readthedocs.io/en/latest/history.html). Changes of each version:
[CHANGELOG.md](https://github.com/Tristanovsk/grstbx/blob/main/CHANGELOG.md).

## Related projects

- [GRS](https://github.com/Tristanovsk/grs): atmospheric and sunglint correction producing the L2A
  images,
- [hGRS](https://github.com/Tristanovsk/hgrs): hyperspectral version of GRS for EnMAP and PRISMA,
- [GRSdriver](https://github.com/Tristanovsk/GRSdriver): readers of the L1C images used by GRS.

## License

GNU General Public License v3, see [LICENSE](https://github.com/Tristanovsk/grstbx/blob/main/LICENSE).
