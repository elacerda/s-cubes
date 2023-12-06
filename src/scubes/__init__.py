import sys
from importlib.metadata import version, metadata

scubes_meta = metadata('s-cubes')
__author__ = scubes_meta['Author-email']
__version__ = version('s-cubes')

def scubes():
    from .utilities.args import scubes_parse_arguments
    from .core import SCubes

    scubes = SCubes(args=scubes_parse_arguments(sys.argv))
    scubes.make(get_mask=True, det_img=True, flam_scale=None)