# S-Cubes

Make galaxy cubes (X, Y, Lambda) with S-PLUS data. This repository include data files to calibrate stamps with iDR4 zero-points correction.

# Requirements

- Python 3.10
- SExtractor installed either locally or system-wide
- `sewpy` [SExtractor python wrapper](https://sewpy.readthedocs.io/en/latest/installation.html) is added to this repository as a git submodule.

# Installation

1. Clone repository with the `sewpy` submodule:

```
> git clone --recurse-submodules https://github.com/cameronmcnz/surface.git
```

2. Install `sewpy` (go to external/sewpy directory):

```
> cd external/sewpy
external/sewpy > pip install .
```

3. Install S-Cubes (go back to s-cubes directory)

```
> pip install .

```

# Example

TODO

# License

This code is distributed under the [GNU GENERAL PUBLIC LICENSE v3.0](LICENSE). Please refer to the `LICENSE` file in the repository for more details.
