GRS products
============

This page summarizes the product layouts that grstbx reads.

.. _file-naming:

File naming
-----------

:class:`~grstbx.datalake.SelectFiles` parses the GRS file names, made of underscore-separated
fields:

.. code-block:: text

   S2A_MSIl2grs_20210906T103021_N0301_R108_T31TGM_20210906T141939_cc004_v14.nc

=========  ==================  ============================================
field      column              content
=========  ==================  ============================================
0          ``satellite``       mission, e.g. ``S2A``
1          ``level``           product level
2          ``date``            acquisition date (index of the file table)
5          ``tile``            tile identifier
7          ``cloud_coverage``  cloud coverage, ``ccXX``
last       ``version``         processor version (file extension removed)
=========  ==================  ============================================

L2A: remote-sensing reflectance
-------------------------------

:meth:`L2grs.load_l2a_image <grstbx.driver.L2grs.load_l2a_image>` supports two layouts:

``*.zarr`` store
   Single-resolution or multiscale (see below). The store is returned as is, without ancillary
   data.

folder ``<name>/``
   Containing ``<name>.nc`` (main image) and ``<name>_anc.nc`` (ancillary data). Products written
   with the ``beam`` metadata profile store one variable per band (``Rrs_<wl>``); they are reshaped
   into a single ``Rrs`` variable with a ``wl`` (wavelength) dimension.

The images are opened lazily with dask chunks of 1000 x 1000 pixels (all wavelengths in one chunk).

L2B: water quality parameters
-----------------------------

netCDF files or Zarr stores (single-resolution or multiscale), with one 2D variable per parameter
(e.g. ``Chla_OC2nasa``).

.. _pyramids:

Multiscale (pyramid) Zarr stores
--------------------------------

Pyramids hold one group per resolution level: ``'0'`` (full resolution), ``'1'`` (2x coarser),
``'2'``, ... The levels are read from the ``multiscales`` attribute of the root group when present,
otherwise from the numbered sub-groups (:func:`~grstbx.driver.zarr_levels`).

.. code-block:: python

   from grstbx import open_zarr_image, zarr_levels

   zarr_levels('S2A_L2B.zarr')                     # ['0', '1', '2', '3']
   ds = open_zarr_image('S2A_L2B.zarr', level=2)    # 4x coarser than full resolution
   ds = open_zarr_image('S2A_L2B.zarr', level=-1)   # coarsest level

The product attributes stored on the root group are added to the dataset of the level, and the
level opened is stored in the ``pyramid_level`` attribute.

Bitmask ``flags``
-----------------

The pixel classification is stored as a single integer ``flags`` raster in which bit ``i`` is set
when condition ``flag_names[i]`` holds. The names and descriptions of the bits are stored in the
``flag_names`` and ``flag_descriptions`` attributes of the variable (as lists, or as
space-separated strings for older products). See :doc:`tutorials/masking`.
