Getting started
===============

**S-Cubes** is a python package designed to make galaxy cubes (X, Y, Lambda) 
with S-PLUS data. The `S-Cubes repository <https://github.com/splus-collab/s-cubes>`_ 
includes the `scubes` python package which allows the users to create their own
codes. It also includes the data files to calibrate stamps with iDR4 zero-points 
correction and SExtractor to identify stars along the stamp (optionally).

.. toctree::
    :maxdepth: 2
    :glob:
    
   
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

   git clone https://github.com/splus-collab/s-cubes.git
   
*(optional)* Create and activate a virtual enviroment for the package
instalation and usage:

.. code:: console

   python3 -m venv .venv
   source .venv/bin/activate

Install S-Cubes:

.. code:: console

   pip install .

.. _scripts:

Entry-points
------------

This package includes various entry-point command-line scripts for 
different tasks. They are: ``scubes``, ``scubesml``, ``get_lupton_RGB``,
``sex_mask_stars``, ``sex_mask_stars_cube``, ``mltoheader`` and ``scubes_filters``.
In order to obtain a detailed description and the script usage run them 
with **–help** argument. See :ref:`How to create a cube` and :ref:`epscripts`.


.. _lic:

License
-------

This code is distributed under the `GNU GENERAL PUBLIC LICENSE
v3.0 <LICENSE>`__. Please refer to the ``LICENSE.txt`` file in the
repository for more details.