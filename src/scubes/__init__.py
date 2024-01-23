from importlib.metadata import version, metadata

from . import data

import os

scubes_meta = metadata('s-cubes')
__author__ = scubes_meta['Author-email']
__version__ = version('s-cubes')

__zp_cat__ = os.path.join(data.__path__[0], 'iDR4_zero-points.csv')
__zpcorr_path__ = os.path.join(data.__path__[0], 'zpcorr_iDR4')