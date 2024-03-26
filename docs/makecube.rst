.. _createcube:

How to create a cube
====================

This example will create a **500x500** pixels cube with the 
12-bands images from **S-PLUS TILE HYDRA-0045** for the *NGC3312* 
galaxy. The fluxes and errors are calculated based on the 
calibration of the zero points of **S-PLUS iDR4** (*data package 
included*), but they are **not corrected for Galactic extinction**.

The stamps are made centered at coordinates RA *10h37m02.5s* and DEC
*-27d33’56"*. See :ref:`radeconv` for more information on how RA and
DEC are parsed to the scripts.

The resultant files will be created at directory *workdir*.

The program also uses SExtractor in order to create a spatial mask of
stars, attempting to remove the areas enclosed by the brightest ones
along the FOV (*-M* optional argument). Do not forget to include the
SExtractor executable path using the option *-x*.

The call to the entry-point script ``scubes`` to this example would be:

.. code:: console

   scubes -I -M -F --w workdir -l 500 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312

See the script help for more options:

.. code:: console

    scubes --help
    
.. toctree::
    :maxdepth: 2
    :glob:
    