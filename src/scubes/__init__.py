from importlib import resources
from importlib.metadata import version, metadata

from . import data

scubes_meta = metadata('s-cubes')
__author__ = scubes_meta['Author-email']
__version__ = version('s-cubes')

__zp_cat__ = str(resources.files(data) / 'iDR4_zero-points.csv')
__zpcorr_path__ = str(resources.files(data) / 'zpcorr_iDR4')