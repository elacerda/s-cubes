import sys
from shutil import which

from .utilities.io import print_level
from .utilities.args import create_parser
from .constants import SPLUS_ARGS, SPLUS_PROG_DESC

def scubes_argparse(args):
    _sex = which(args.sextractor)
    if _sex is None:
        print_level(f'{args.sextractor}: SExtractor exec not found', 1, args.verbose)
        _SExtr_names = ['sex', 'source-extractor']
        for name in _SExtr_names:
            _sex = which(name)
            if _sex is None:
                print_level(f'{name}: SExtractor exec not found', 2, args.verbose)
            else:
                print_level(f'{name}: SExtractor found. Forcing --sextractor={_sex}', 1, args.verbose)
                args.sextractor = _sex
                pass
        if _sex is None:
            print_level(f'SExtractor not found')
            sys.exit(1)
    return args

def scubes():
    from .core import SCubes

    parser = create_parser(args_dict=SPLUS_ARGS, program_description=SPLUS_PROG_DESC)
    args = scubes_argparse(parser.parse_args(args=sys.argv[1:]))
    scubes = SCubes(args)
    scubes.create_cube(flam_scale=None)