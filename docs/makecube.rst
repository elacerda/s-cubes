How to create a cube
====================

This example will create a **500x500** pixels cube with the 
12-bands images from **S-PLUS TILE HYDRA-0045** for the *NGC3312* 
galaxy. The fluxes and errors are calculated based on the 
calibration of the zero points of **S-PLUS iDR4** (*data package 
included*), but they are **not corrected for Galactic extinction**.

The stamps are made centered at coordinates RA *10h37m02.5s* and DEC
*-27d33’56"*. Examples of how RA and DEC will be parsed by the code:

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

   scubes -I -M -F --w workdir -l 500 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312

See the script help for more options:

.. code:: console

    scubes --help
    