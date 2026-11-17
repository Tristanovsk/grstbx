"""
Lazy ``.rio`` accessor: rioxarray (about 1.2 s to import, with rasterio and dask.array) is imported on the first use
of ``.rio`` on an xarray object, not when grstbx is imported. Its own accessors then replace this one.
"""

import warnings

import xarray as xr


def _lazy(kind):
    class _LazyRio:
        def __new__(cls, obj):
            import rioxarray  # noqa: F401, registers the real accessors over this one
            accessor = rioxarray.raster_dataset.RasterDataset if kind == 'dataset' else \
                rioxarray.raster_array.RasterArray
            return accessor(obj)
    return _LazyRio


def register():
    """Register the lazy accessors, unless rioxarray is already imported (its accessors are kept)."""
    import sys

    if 'rioxarray' in sys.modules:
        return
    # rioxarray, imported later (here or by the user), overrides these accessors: expected, not worth a warning
    warnings.filterwarnings('ignore', message=r".*accessor 'rio'.*",
                            category=xr.core.extensions.AccessorRegistrationWarning)
    xr.register_dataset_accessor('rio')(_lazy('dataset'))
    xr.register_dataarray_accessor('rio')(_lazy('dataarray'))
