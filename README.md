# **grstbx**
## **Scientific code and notebooks to visualize and post-process GRS L2A and L2B images** 


## Example
![example gif](illustration/grstbx_visual_tool.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Please use conda environment
``` 
conda activate "name of your conda env"
```

Python >= 3.9 is recommended, example:
``` 
conda create python=3.11 -n grstbx
conda activate grstbx
``` 

It is safer to pre-install gdal and pyproj with conda:

```
conda install -c conda-forge gdal pyproj
```

### Installing

Clone [the repository](https://github.com/Tristanovsk/grstbx#) and execute the following command in the
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