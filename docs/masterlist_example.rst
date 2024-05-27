.. _jupyter_masterlist:

Masterlist run example
======================

:download:`Download this notebook <masterlist_example.ipynb>`.

.. toctree::
    :maxdepth: 2
    :glob:

``scubes`` package implements a script to run the program from a CSV
file with a list of objects, a *Masterlist*. This script is called
``scubesml``

*Masterlist* is a csv text file in which one could gather information of
a list of objects to create cubes. The file must contain at least 5
columns named with the following header and information:

1. ``SNAME``: A nickname for the object
2. ``FIELD``: S-PLUS Field (TILE) in which the program will search for
   the coordinates
3. ``RA__deg``: Right-ascencion in degrees
4. ``DEC__deg``: Declination in degrees
5. ``SIZE__pix``: SIZE of the object in pixels

*Masterlist* file content example::

::

   SNAME,FIELD,RA__deg,DEC__deg,SIZE__pix
   S00001,SPLUS-s24s34,52.08196,-31.06817,53.65902
   S00002,SPLUS-s24s35,52.87771,-30.21333,25.898617
   (...)

A *Masterlist* could contain more columns and, at the end of the run,
the script will update the primary header of the output FITS file with
all information inside the *Masterlist* for the chosen object.

.. code-block:: ipython3

    !scubesml --help


.. parsed-literal::

    usage: scubesml [-h] [-r] [-c] [-f] [-b BANDS [BANDS ...]]
                    [-S SIZE_MULTIPLICATOR] [-w WORK_DIR] [-o OUTPUT_DIR] [-v]
                    [-D] [-Z ZPCORR_DIR] [-z ZP_TABLE] [-U USERNAME] [-P PASSWORD]
                    [-R] [--version]
                    GALAXY_SNAME MASTERLIST
    
    ┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐  | scubesml entry-point script:
    └─┐───│  │ │├┴┐├┤ └─┐  | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES
    └─┘   └─┘└─┘└─┘└─┘└─┘  | using the masterlist information as input.
    ---------------------- + 
    
       Eduardo Alberto Duarte Lacerda <dhubax@gmail.com>, Fabio Herpich <fabiorafaelh@gmail.com>
    
    positional arguments:
      GALAXY_SNAME                Galaxy's masterlist nickname
      MASTERLIST                  Path to masterlist file
    
    options:
      -h, --help                  show this help message and exit
      -r, --redo                  Enable redo mode to overwrite final cubes.
                                  Default value is False
      -c, --clean                 Clean intermediate files after processing.
                                  Default value is False
      -f, --force                 Force overwrite of existing files. Default value
                                  is False
      -b BANDS [BANDS ...], --bands BANDS [BANDS ...]
                                  List of S-PLUS bands (space separated). Default
                                  value is ['U', 'F378', 'F395', 'F410', 'F430',
                                  'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
      -S SIZE_MULTIPLICATOR, --size_multiplicator SIZE_MULTIPLICATOR
                                  Factor to multiply the SIZE__pix value of the
                                  masterlist to create the galaxy size. If size is
                                  a odd number, the program will choose the
                                  closest even integer. Default value is 10
      -w WORK_DIR, --work_dir WORK_DIR
                                  Working directory. Default value is /storage/hdd
                                  /backup/dhubax/dev/astro/splus/s-cubes/workdir
      -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                                  Output directory. Default value is /storage/hdd/
                                  backup/dhubax/dev/astro/splus/s-cubes/workdir
      -v, --verbose               Verbosity level.
      -D, --debug                 Enable debug mode. Default value is False
      -Z ZPCORR_DIR, --zpcorr_dir ZPCORR_DIR
                                  Zero-point correction directory. Default value
                                  is /home/lacerda/.local/lib/python3.10/site-
                                  packages/scubes/data/zpcorr_iDR4
      -z ZP_TABLE, --zp_table ZP_TABLE
                                  Zero-point table. Default value is
                                  /home/lacerda/.local/lib/python3.10/site-
                                  packages/scubes/data/iDR4_zero-points.csv
      -U USERNAME, --username USERNAME
                                  S-PLUS Cloud username.
      -P PASSWORD, --password PASSWORD
                                  S-PLUS Cloud password.
      -R, --remove_downloaded_data
                                  Remove the downloaded data from splusdata at the
                                  end of the run. Default value is False
      --version                   show program's version number and exit


Size of the stamp
-----------------

For the size calculation, the script will use the value of the column
``SIZE__pix`` and also the ``--size_multiplicator`` option. At the end,
the final size value will be the next even integer of the
multiplication:

::

   size = size_multiplicator x SIZE__pix

.. code-block:: ipython3

    !cat masterlist_example.csv


.. parsed-literal::

    SNAME,FIELD,RA__deg,DEC__deg,TYPE,VELOCITY__kms,REDSHIFT,DISTANCE__Mpc,EBV__mag,SIZE__pix
    NGC1344,SPLUS-s24s34,52.08196,-31.06817,G,1241.0,0.00414,18.364103654698095,0.0158,53.65902
    ESO418-G008,SPLUS-s24s35,52.87771,-30.21333,G,1195.0,0.003987,17.683362136194148,0.0134,25.898617
    ESO418-G009,SPLUS-s24s35,52.98173,-31.33763,G,972.0,0.003242,14.370905842817514,0.0172,31.746374
    NGC1366,SPLUS-s24s35,53.47367,-31.19411,G,1231.0,0.004106,18.212814064954173,0.0146,15.589648
    NGC1406,SPLUS-s24s36,54.84708,-31.32142,G,1075.0,0.003585,15.895502593028134,0.0094,53.065895
    ESO419-G013,SPLUS-s24s39,60.17338,-30.831,G,1490.0,0.00497,22.059776288801466,0.0064,23.614521
    ESO420-G009,SPLUS-s24s41,62.75269,-31.40743,G,1367.0,0.00456,20.23362042333429,0.0197,35.48054
    IC1913,SPLUS-s25s32,49.8939,-32.46502,G,1443.0,0.004813,21.360357850834735,0.0138,28.049528
    IC1919,SPLUS-s25s33,51.50933,-32.89456,G,1323.0,0.004413,19.579153744085982,0.0125,37.4627


The *Masterlist* file ``masterlist_example.csv`` contain 9 objects. To
run ``scubes`` for each object, just use ``scubesml`` script as:

.. code-block:: ipython3

    !scubesml -frR -U YOURUSER -P YOURPASS -w . --size_multiplicator 20 -- ESO419-G013 masterlist_example.csv


.. parsed-literal::

    ESO419-G013 @ SPLUS-s24s39 - downloading: 100%|█| 12/12 [00:27<00:00,  2.29s/it]
    [0;33mWARNING[0m: FITSFixedWarning: 'datfix' made the change 'Set DATE-OBS to '2017-09-26' from MJD-OBS'. [astropy.wcs.wcs]
    [0;33mWARNING[0m: FITSFixedWarning: 'datfix' made the change 'Set DATE-OBS to '2017-10-13' from MJD-OBS'. [astropy.wcs.wcs]
    [2024-05-26T21:34:22.521107] - scubesml: Reading ZPs table: /home/lacerda/.local/lib/python3.10/site-packages/scubes/data/iDR4_zero-points.csv
    [2024-05-26T21:34:22.524050] - scubesml: Getting ZP corrections for the S-PLUS bands...
    [2024-05-26T21:34:22.528228] - scubesml: Calibrating stamps...
    /home/lacerda/.local/lib/python3.10/site-packages/scubes/core.py:523: RuntimeWarning: cdelt will be ignored since cd is present
      nw.wcs.cdelt[:2] = w.wcs.cdelt
    [2024-05-26T21:34:23.287041] - scubesml: Cube successfully created!
    [2024-05-26T21:34:23.287061] - scubesml: Removing downloaded data


Header information
------------------

At this point, we can see that the *Masterlist* information is stored at
the header:

.. code-block:: ipython3

    from scubes.utilities.readscube import read_scube
    
    filename = 'ESO419-G013/ESO419-G013_cube.fits'
    scube = read_scube(filename)
    scube.primary_header




.. parsed-literal::

    SIMPLE  =                    T / conforms to FITS standard                      
    BITPIX  =                    8 / array data type                                
    NAXIS   =                    0 / number of array dimensions                     
    EXTEND  =                    T                                                  
    TILE    = 'SPLUS-s24s39'                                                        
    GALAXY  = 'ESO419-G013'                                                         
    SIZE    =                  472 / Side of the stamp in pixels                    
    X0TILE  =             3053.381                                                  
    Y0TILE  =             5962.645                                                  
    RA      =             60.17338 / deg                                            
    DEC     =              -30.831 / deg                                            
    TYPE    = 'G       '                                                            
    VELOCITY=               1490.0 / kms                                            
    REDSHIFT=              0.00497                                                  
    DISTANCE=   22.059776288801466 / Mpc                                            
    EBV     =               0.0064 / mag                                            
    SIZE_ML =            23.614521 / SIZE masterlist                                


Running scubes for the entire list
----------------------------------

A simple shell script could help to run ``scubes`` for the entire list
of objects. We use ``tail`` and ``cut`` shell commands in order to get
only the SNAME of the objects from the *Masterlist* file.

.. code-block:: bash

   for SNAME in `tail -n+2 masterlist_example.csv | cut -f1 -d','`
   do
       scubesml -frR -U YOURUSER -P YOURPASS --size_multiplicator 20 -- ${SNAME} masterlist_example.csv
   done
