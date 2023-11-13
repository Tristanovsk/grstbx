# **grstbx**
## **Scientific code to visualize and post-process GRS L2 images** 
## --> remote sensing reflectance (Rrs) 

## Example
![example gif](illustration/grstbx_visual_tool.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
conda install -c conda-forge gdal eoreader pyproj rasterio datashader cartopy hvplot jupyter jupyterlab jupyter_bokeh dask rioxarray
```

### Installing

First, clone [the repository](https://github.com/Tristanovsk/invRrs#) and execute the following command in the
local copy:

```
pip install .
```

## Some example of the forward model:

![example files](illustration/le_leman_bleu.png)

## To generate a new kernel for Jupyter
```
python -m ipykernel install --user --name=grstbx
```