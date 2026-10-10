.. _api:

API reference
=============

The main classes and functions are importable from the package itself:

.. code-block:: python

   import grstbx

   select = grstbx.SelectFiles('/datalake/watcal')
   dc = grstbx.L2grs(files)
   masking_ = grstbx.Masking(dc.datacube)
   subset, subset_anc = grstbx.crop_l2a(product, aoi, ancillary)
   viewer = grstbx.visual.ViewSpectral(dc.datacube.Rrs)

File selection
--------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   grstbx.datalake.SelectFiles

Loading images and datacubes
----------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   grstbx.driver.L2grs

.. autosummary::
   :toctree: generated
   :nosignatures:

   grstbx.driver.open_zarr_image
   grstbx.driver.open_zarr_ancillary
   grstbx.driver.zarr_levels
   grstbx.driver.is_zarr

Cropping and export
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   grstbx.subset.open_l2a
   grstbx.subset.aoi_from_box
   grstbx.subset.crop_l2a
   grstbx.subset.export_l2a
   grstbx.subset.export_rrs_geotiff
   grstbx.subset.subset_path
   grstbx.subset.crop_and_export

Masking
-------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   grstbx.masking.Masking

Interactive visualization
-------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   grstbx.visual.ViewSpectral
   grstbx.visual.ViewParam
   grstbx.visual.Multiscale
   grstbx.visual.ImageViewer
   grstbx.visual.Utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   grstbx.visual.basemap_tiles
   grstbx.visual.set_stadia_api_key
   grstbx.visual.get_colormap

The basemaps proposed by the viewers (``grstbx.visual.BASEMAPS``) are free XYZ tile services: CARTO
(Positron, Dark Matter, Voyager), OpenStreetMap, OpenTopoMap, Esri (World Imagery, World Topo, Ocean)
and the Stamen / Alidade styles of Stadia Maps. Stadia Maps tiles are free without key when the
notebook runs on ``localhost``; otherwise create a free account and set its API key with
:func:`grstbx.visual.set_stadia_api_key` or the ``STADIA_API_KEY`` environment variable.

The single bands of :class:`~grstbx.visual.ViewSpectral` are drawn with sequential, perceptually
uniform colormaps (``grstbx.visual.RRS_COLORMAPS``) named after the water feature they enhance:
rainbow (CET_R1, default), turbid water and sediments (cmocean *turbid*), clear / shallow water
(cmocean *deep*), algae (cmocean *algae*), water masses (cmocean *haline*), Gouldian, viridis,
cividis (colour-blind safe), blues, fire and grey; they can be reversed in the widget and the
initial one is set with ``cmap=`` (label, colorcet, ``'cmo.<name>'`` or matplotlib name).

Utilities
---------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   grstbx.utils.SpatioTemp
   grstbx.utils.Data
   grstbx.utils.Irradiance
   grstbx.utils.Plotting
   grstbx.utils.Dem
   grstbx.utils.Reproj

.. autosummary::
   :toctree: generated
   :nosignatures:

   grstbx.utils.utm_projection
   grstbx.utils.plot_rgb

Module overviews
----------------

.. toctree::
   :maxdepth: 1

   api/modules
