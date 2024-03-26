.. _epscripts:

Entry-point scripts
===================

The scubes package includes various entry-point command-line scripts for 
different tasks. They are: :ref:`epscubes`, :ref:`epscubesml`, :ref:`epgetluptonrgb`,
:ref:`epsexmaskstars`, :ref:`epsexmaskstarscube`, :ref:`epmltoheader`. Scripts 
with the ``ml`` (master list) identificator are optimized to work with 
information from with a list of objects used as input. The *cube* suffix will identify
scripts which work with a cube produced by :ref:`epscubes` or :ref:`epscubesml` 
as input.

All scripts include the ``--help`` option for more information.

.. toctree::
    :maxdepth: 2
    :glob:
    
.. _radeconv:

Right-ascencion and Declination conversion
------------------------------------------

The input values of RA and DEC will be converted to degrees using the 
:meth:`scubes.utilities.io.convert_coord_to_degrees`. All scripts with RA 
and DEC inputs parse angles in two different units:

- **hourangle**: using *hms* divisors; Ex: *10h37m2.5s*
- **degrees**: using *:* or *dms*  divisors; Ex: *10:37:2.5* or *10d37m2.5s*

Note that *10h37m2.5s* is a totally different angle from *10:37:2.5* 
(*159.26 deg* and *10.62 deg* respectively).

.. _epscubes:

``scubes`` 
----------

:meth:`scubes.entry_points.scubes` is the main script of **S-Cubes**. It 
calibrates the zero-points, calculates the fluxes and uncertainties for the 
12-band images cropped from `S-PLUS <https://www.splus.iag.usp.br/>`__ 
observed tiles. The stamps ared downloaded from 
`S-PLUS Cloud <https://splus.cloud/>`__ and the zero-points for the 
data-release 4 (DR4) are 
`hard-coded <https://github.com/elacerda/s-cubes/tree/main/src/scubes/data>`__ 
on the package.

The usage of this script is detailed in :ref:`How to create a cube`.

.. _epscubesml:

``scubesml`` 
------------
xxx

.. _epgetluptonrgb:

``get_lupton_RGB`` 
------------------
xxx

.. _epsexmaskstars:

``sex_mask_stars``
------------------
xxx

.. _epsexmaskstarscube:

``sex_mask_stars_cube`` 
-----------------------
xxx

.. _epmltoheader:

``mltoheader``
--------------
xxx
