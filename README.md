# S-Cubes

Make galaxy cubes (X, Y, Lambda) with S-PLUS data. This repository include data files to calibrate stamps with iDR4 zero-points correction and SExtractor to identify stars along the stamp (optionally).

# Requirements

- Python 3.10
- SExtractor installed either locally or system-wide
- `sewpy` [SExtractor python wrapper](https://github.com/megalut/sewpy) is added to this repository as a git submodule.
- An user account at [S-PLUS Cloud](https://splus.cloud/) in order to access the necessary data.

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
Type `scubes --help` for help and usage.

# Entry-point scripts

This package includes three entry-point command-line scripts: `scubes`, `sex_mask_stars` and `get_lupton_rgb`. In order to obtain a detailed description and the script usage run them with --help argument.

# `scubes` Running example

This example will create a 500x500 pixels cube (at redshift=0.009627) with the 12-bands images from S-PLUS TILE HYDRA-0045 for the NGC3312 galaxy. The fluxes and errors are calculated based on the calibration of the zero points of S-PLUS iDR4 (package included). The stamps are made centered at coordinates RA 10h37m02.5s and DEC -27d33'56". The resultant files will be created at directory `workdir`.

The call to the entry-point script `scubes` to this example would be:

```console
scubes -I -M -F --w workdir -l 500 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312 0.009627
```

# License

This code is distributed under the [GNU GENERAL PUBLIC LICENSE v3.0](LICENSE). Please refer to the `LICENSE.txt` file in the repository for more details.
