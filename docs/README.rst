README
======

Make galaxy cubes (X, Y, Lambda) with S-PLUS data. This repository
include data files to calibrate stamps with iDR4 zero-points correction
and SExtractor to identify stars along the stamp (optionally).

Requirements
------------

-  Python 3.10
-  **SExtractor** installed either *locally* or *system-wide*
-  ``sewpy`` `SExtractor python
   wrapper <https://github.com/megalut/sewpy>`__ is added to this
   repository as a git submodule.
-  An user account at `S-PLUS Cloud <https://splus.cloud/>`__ in order
   to access the necessary data.

Installation
------------

Clone repository with the ``sewpy`` submodule:

.. code:: console

   git clone --recurse-submodules https://github.com/elacerda/s-cubes.git
   cd s-cubes

Create and activate a virtual enviroment for the package instalation and
usage:

.. code:: console

   python3 -m venv .venv
   source .venv/bin/activate

Install ``sewpy``:

.. code:: console

   cd external/sewpy && pip3 install . && cd -

Install S-Cubes (go back to s-cubes directory)

.. code:: console

   pip3 install .

Entry-point scripts
-------------------

This package includes three entry-point command-line scripts:
``scubes``, ``sex_mask_stars`` and ``get_lupton_RGB``. In order to
obtain a detailed description and the script usage run them with
**--help** argument.

``scubes`` Running example
--------------------------

This example will create a **500x500** pixels cube (at
redshift=\ *0.009627*) with the 12-bands images from **S-PLUS TILE
HYDRA-0045** for the *NGC3312* galaxy. The fluxes and errors are
calculated based on the calibration of the zero points of **S-PLUS
iDR4** (*data package included*), but they are **not corrected for 
Galactic extinction**.

The stamps are made centered at coordinates RA *10h37m02.5s* and DEC
*-27d33'56"*. Examples of how RA and DEC will be parsed by the code:

::

   RA=03h28m19.59s                 RA(parsed)=03h28m19.59s
   DEC=-31d04m05.26392275s         DEC(parsed)=-31d04m05.26392275s
   astropy.coordinates.SkyCoord() object:
   <SkyCoord (ICRS): (ra, dec) in deg
       (52.081625, -31.06812887)>

   RA=03:28:19.59                  RA(parsed)=03:28:19.59°
   DEC=-31:04:05.26392275          DEC(parsed)=-31:04:05.26392275°
   astropy.coordinates.SkyCoord() object:
   <SkyCoord (ICRS): (ra, dec) in deg
       (3.47210833, -31.06812887)>

   RA=03d28'19.59"                 RA(parsed)=03d28'19.59"
   DEC=-31d04'05.26392275"         DEC(parsed)=-31d04'05.26392275"
   astropy.coordinates.SkyCoord() object:
   <SkyCoord (ICRS): (ra, dec) in deg
       (3.47210833, -31.06812887)>

The resultant files will be created at directory *workdir*.

The program also uses SExtractor in order to create a spatial mask of
stars, attempting to remove the areas enclosed by the brightest ones
along the FOV (*-M* optional argument). Do not forget to include the
SExtractor executable path using the option *-x*.

The call to the entry-point script ``scubes`` to this example would be:

.. code:: console

   scubes -I -M -F --w workdir -l 500 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312 0.009627

License
-------

This code is distributed under the `GNU GENERAL PUBLIC LICENSE
v3.0 <LICENSE>`__. Please refer to the ``LICENSE.txt`` file in the
repository for more details.