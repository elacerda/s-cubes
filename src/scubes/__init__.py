import sys
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('s-cubes')
except PackageNotFoundError as e:
    pass

def scubes():
    from .utilities.args import scubes_parse_arguments
    from .core import SCubes

    scubes = SCubes(args=scubes_parse_arguments(sys.argv))
    scubes.make(get_mask=True, det_img=True, flam_scale=None)