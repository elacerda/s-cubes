import os

from astropy.io import ascii
from importlib.metadata import version, metadata

from . import data

scubes_meta = metadata('s-cubes')
__author__ = scubes_meta['Author-email']
__version__ = version('s-cubes')

__dr4_zp_cat__ = os.path.join(data.__path__[0], 'iDR4_zero-points.csv')
__dr4_zpcorr_path__ = os.path.join(data.__path__[0], 'zpcorr_iDR4')
__dr5_zp_cat__ = os.path.join(data.__path__[0], 'iDR5_fields_zps.csv')
__dr5_mar_zpcorr_path__ = os.path.join(data.__path__[0], 'zpcorr_iDR5', 'mar')
__dr5_jyp_zpcorr_path__ = os.path.join(data.__path__[0], 'zpcorr_iDR5', 'jype')

__dr5_zpcorr__ = {'mar': __dr5_mar_zpcorr_path__, 'jype': __dr5_jyp_zpcorr_path__}

__filters_table__ = ascii.read(os.path.join(data.__path__[0], 'central_wavelengths.csv'))