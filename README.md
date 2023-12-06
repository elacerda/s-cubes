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
cd s-cubes
```

Create and activate a virtual enviroment for the package instalation and usage:

```console
python3 -m venv .venv
source .venv/bin/activate
```

Install `sewpy`:

```console
pip3 install external/sewpy/.
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

┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐  | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES. 
└─┐───│  │ │├┴┐├┤ └─┐  | S-CUBES is an organized FITS file with data, errors, 
└─┘   └─┘└─┘└─┘└─┘└─┘  | mask and metadata about some galaxy present on any 
---------------------- + S-PLUS observed tile. Any problem contact:

                Eduardo A. D. Lacerda - dhubax@gmail.com

                (...)
```

# Running example

TODO

# License

This code is distributed under the [GNU GENERAL PUBLIC LICENSE v3.0](LICENSE). Please refer to the `LICENSE.txt` file in the repository for more details.
