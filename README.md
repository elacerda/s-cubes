# S-Cubes

Make galaxy cubes (X, Y, Lambda) with S-PLUS data. This repository include data files to calibrate stamps with iDR4 zero-points correction.

# Requirements

- Python 3.10
- SExtractor installed either locally or system-wide
- `sewpy` [SExtractor python wrapper](https://sewpy.readthedocs.io/en/latest/installation.html) is added to this repository as a git submodule.

# Installation

Clone repository with the `sewpy` submodule:

```console
git clone --recurse-submodules https://github.com/elacerda/s-cubes.git
```

Create and activate a virtual enviroment for the package instalation and usage:

```console
python3 -m venv .venv
source .venv/bin/activate
```

Install `sewpy` (go to external/sewpy directory):

```console
cd external/sewpy
external/sewpy > pip3 install .
```

Install S-Cubes (go back to s-cubes directory)

```console
pip3 install .

```
Type `scubes --help`:

```console
usage: scubes [-h] [-r] [-c] [-f] [-b BANDS] [-l SIZE] [-a ANGSIZE] [-w WORK_DIR] 
              [-o OUTPUT_DIR] [-x SEXTRACTOR] [-p CLASS_STAR] [-v] [--debug] 
              [--satur_level SATUR_LEVEL] [--data_dir DATA_DIR] [--zpcorr_dir ZPCORR_DIR] 
              [--zp_table ZP_TABLE] [--back_size BACK_SIZE] [--detect_thresh DETECT_THRESH]
              SPLUS_TILE RA DEC GALAXY_NAME REDSHIFT

┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐  | Create S-PLUS galaxies data cubes, a.k.a. SCUBES. 
└─┐───│  │ │├┴┐├┤ └─┐  | SCUBES is an organized FITS file with data, errors, 
└─┘   └─┘└─┘└─┘└─┘└─┘  | mask and metadata about some galaxy present on any 
---------------------- + S-PLUS observed tile. Any problem contact:

                Eduardo A. D. Lacerda - dhubax@gmail.com

positional arguments:
  SPLUS_TILE                  Name of the S-PLUS tile
  RA                          Galaxy's right ascension
  DEC                         Galaxy's declination
  GALAXY_NAME                 Galaxy's name
  REDSHIFT                    Spectroscopic or photometric redshift of the galaxy

options:
  -h, --help                  show this help message and exit
  -r, --redo                  Enable redo mode to overwrite final cubes
  -c, --clean                 Clean intermediate files after processing
  -f, --force                 Force overwrite of existing files
  -b BANDS, --bands BANDS     List of S-PLUS bands
  -l SIZE, --size SIZE        Size of the cube in pixels
  -a ANGSIZE, --angsize ANGSIZE
                              Galaxy's Angular size in arcsec
  -w WORK_DIR, --work_dir WORK_DIR
                              Working directory
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                              Output directory
  -x SEXTRACTOR, --sextractor SEXTRACTOR
                              Path to SExtractor executable
  -p CLASS_STAR, --class_star CLASS_STAR
                              SExtractor CLASS_STAR parameter for star/galaxy separation
  -v, --verbose
  --debug                     Enable debug mode
  --satur_level SATUR_LEVEL   Saturation level for the png images. Default is 1600.0
  --data_dir DATA_DIR         Data directory
  --zpcorr_dir ZPCORR_DIR     Zero-point correction directory
  --zp_table ZP_TABLE         Zero-point table
  --back_size BACK_SIZE       Background mesh size for SExtractor. Default is 64
  --detect_thresh DETECT_THRESH
                              Detection threshold for SExtractor. Default is 1.1
```

# Running example

TODO

# License

This code is distributed under the [GNU GENERAL PUBLIC LICENSE v3.0](LICENSE). Please refer to the `LICENSE.txt` file in the repository for more details.
