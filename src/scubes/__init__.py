import sys
from shutil import which
from os.path import basename, dirname, realpath
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('s-cubes')
except PackageNotFoundError as e:
    pass

__script_name__ = basename(sys.argv[0])

from .utilities.io import print_level
from .utilities.args import create_parser
from .core import SCubes


def scubes_parse_arguments():
    parser = create_parser()
    if len(sys.argv) == 1:
        print_level(f'{__script_name__}: missing arguments', 0, 1)
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(args=sys.argv[1:])
    _sex = which(args.sextractor)
    if _sex is None:
        print_level(f'{__script_name__}: {args.sextractor}: SExtractor exec not found', 2, args.verbose)
        _SExtr_names = ['sex', 'source-extractor']
        for name in _SExtr_names:
            _sex = which(name)
            if _sex is None:
                print_level(f'{__script_name__}: {name}: SExtractor exec not found', 2, args.verbose)
            else:
                args.sextractor = _sex
                pass
        if _sex is None:
            print_level(f'{__script_name__}: SExtractor not found')
            sys.exit(1)
    return args

def scubes():
    scubes = SCubes(args=scubes_parse_arguments(), program_name=__script_name__)
    scubes.make(get_mask=True, det_img=True, flam_scale=None)