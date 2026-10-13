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

Matchups
--------

Pairing of satellite and in situ data, agreement statistics and scatter plots (see the L2B
validation tutorial).

.. autosummary::
   :toctree: generated
   :nosignatures:

   grstbx.matchup.get_matchups
   grstbx.matchup.matchup_metrics
   grstbx.matchup.plot_matchup

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
and the Stamen / Alidade styles of Stadia Maps. The default, OpenStreetMap, needs no key. Stadia
Maps tiles are free without key only when the notebook runs on ``localhost`` (elsewhere, such as in
a published page, they return error 401); otherwise create a free account and set its API key with
:func:`grstbx.visual.set_stadia_api_key` or the ``STADIA_API_KEY`` environment variable.

The single bands of :class:`~grstbx.visual.ViewSpectral` and the parameters of
:class:`~grstbx.visual.ViewParam` are drawn with sequential, perceptually uniform colormaps
(``grstbx.visual.RRS_COLORMAPS``) named after the water feature they enhance:
rainbow (CET_R1, default), turbid water and sediments (cmocean *turbid*), clear / shallow water
(cmocean *deep*), algae (cmocean *algae*), water masses (cmocean *haline*), Gouldian, viridis,
cividis (colour-blind safe), blues, fire and grey; they can be reversed in the widget and the
initial one is set with ``cmap=`` (label, colorcet, ``'cmo.<name>'`` or matplotlib name). For L2B
parameters better shown with a diverging map, pass it with ``cmap=`` (e.g. ``cmap='CET_D13'``): it is
added to the list.

:class:`~grstbx.visual.ViewSpectral` plots live the full-resolution spectrum of the pixel under the
mouse (*Live spectrum* checkbox), next to the spectra of the drawn points. The block of pixels around
the mouse, aligned on the Zarr chunks, is read once with all the bands and kept in memory
(:meth:`grstbx.visual.Multiscale.pixel`), so that hovering nearby does not read the data again.

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
