S-Cubes
=======

.. image:: https://img.shields.io/github/v/tag/elacerda/s-cubes?include_prereleases&label=version
   :target: https://github.com/elacerda/s-cubes
   :alt: GitHub Tag
.. image:: https://img.shields.io/github/actions/workflow/status/elacerda/s-cubes/.github%2Fworkflows%2Fdocs_deploy.yml?label=gh-pages
   :target: https://github.com/elacerda/s-cubes/blob/main/.github/workflows/docs_deploy.yml 
   :alt: GitHub Actions Workflow Status 
.. image:: https://img.shields.io/github/license/elacerda/s-cubes
   :target: https://github.com/elacerda/s-cubes/blob/main/LICENSE.txt
   :alt: GitHub License
.. image:: https://img.shields.io/website?url=https%3A%2F%2Felacerda.github.io%2Fs-cubes%2F
   :target: https://elacerda.github.io/s-cubes/index.html
   :alt: Website

S-Cubes is a python package designed to make galaxy cubes (X, Y, Lambda) with S-PLUS data. 
The S-Cubes repository includes the scubes python package which allows the users to create 
their own codes. It also includes the data files to calibrate stamps with iDR4 zero-points 
correction and SExtractor to identify stars along the stamp (optionally).

**Online Documentation:** `<https://elacerda.github.io/s-cubes/>`__

.. _require:

Requirements
------------

-  Python 3.8
-  **SExtractor** installed either *locally* or *system-wide*
-  An user account at `S-PLUS Cloud <https://splus.cloud/>`__ in order
   to access the necessary data.

.. _install:

Installation
------------

Clone the project:

.. code:: console

   git clone https://github.com/elacerda/s-cubes.git
   
*(optional)* Create and activate a virtual enviroment for the package
instalation and usage:

.. code:: console

   python3 -m venv .venv
   source .venv/bin/activate

Install S-Cubes:

.. code:: console

   pip3 install .

.. _scripts:

Entry-point scripts
-------------------

This package includes various entry-point command-line scripts for 
different tasks. They are: ``scubes``, ``scubesml``, ``get_lupton_RGB``,
``sex_mask_stars``, ``sex_mask_stars_cube``, ``mltoheader`` and
``scubes_filters``. In order to obtain a detailed description and the 
script usage run them with **–help** argument. 

.. _example:

``scubes`` Running example
..........................

This example will create a **500x500** pixels cube with the 
12-bands images from **S-PLUS TILE HYDRA-0045** for the *NGC3312* 
galaxy. The fluxes and errors are calculated based on the 
calibration of the zero points of **S-PLUS iDR4** (*data package 
included*), but they are **not corrected for Galactic extinction**.

The stamps are made centered at coordinates RA *10h37m02.5s* and DEC
*-27d33’56"*. The input values of RA and DEC will be converted to 
degrees using the :meth:`scubes.utilities.io.convert_coord_to_degrees`. 
It accepts hourangle (using *hms* divisors, ex: *10h37m2.5s*) and 
degrees (using *:* or *dms* divisords, ex: *10:37:2.5* or *10d37m2.5s*).
Note that *10h37m2.5s* (*159.26 deg*) is a totally different angle from 
*10:37:2.5* (*10.62 deg*).

The resultant files will be created at directory *workdir*.

The program also uses SExtractor in order to create a spatial mask of
stars, attempting to remove the areas enclosed by the brightest ones
along the FOV (*-M* optional argument). Do not forget to include the
SExtractor executable path using the option *-x*.

The call to the entry-point script ``scubes`` to this example would be:

.. code:: console

   scubes -I -M -F --w workdir -l 500 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312

.. _license:

License
-------

This code is distributed under the `GNU GENERAL PUBLIC LICENSE
v3.0 <LICENSE>`__. Please refer to the ``LICENSE.txt`` file in the
repository for more details.