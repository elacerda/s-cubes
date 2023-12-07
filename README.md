# S-Cubes

Make galaxy cubes (X, Y, Lambda) with S-PLUS data. This repository include data files to calibrate stamps with iDR4 zero-points correction.

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

# Running example

This example will create a 500x500 pixels cube (at redshift=0.009627) with the 12-bands images from S-PLUS TILE HYDRA-0045 for the NGC3312 galaxy. The fluxes and errors are calculated based on the calibration of the zero points of S-PLUS iDR4. The stamps are made centered at coordinates RA 10h37m02.5s and DEC -27d33'56". The resultant files will be created at directory `workdir` and the program will retrieve the data from directory data, with a tree like this one:

```console
data
├── iDR4_zero-points.csv
├── sex_data
│   └── tophat_3.0_3x3.conv
└── zpcorr_iDR4
    ├── SPLUS_F378_offsets_grid.npy
    ├── SPLUS_F395_offsets_grid.npy
    ├── SPLUS_F410_offsets_grid.npy
    ├── SPLUS_F430_offsets_grid.npy
    ├── SPLUS_F515_offsets_grid.npy
    ├── SPLUS_F660_offsets_grid.npy
    ├── SPLUS_F861_offsets_grid.npy
    ├── SPLUS_G_offsets_grid.npy
    ├── SPLUS_I_offsets_grid.npy
    ├── SPLUS_R_offsets_grid.npy
    ├── SPLUS_U_offsets_grid.npy
    └── SPLUS_Z_offsets_grid.npy
```

The call to the entry-point script `scubes` to this example would be:

```console
scubes --data_dir data --zpcorr_dir zpcorr_iDR4 --zp_table iDR4_zero-points.csv --w workdir -l 500 -a 110 -x /usr/bin/source-extractor -- HYDRA-0045 10h37m02.5s -27d33\'56\" NGC3312 0.009627
```

# License

This code is distributed under the [GNU GENERAL PUBLIC LICENSE v3.0](LICENSE). Please refer to the `LICENSE.txt` file in the repository for more details.
