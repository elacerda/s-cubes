import sys
from os import getcwd
from shutil import which
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter

from .constants import SPLUS_PROG_DESC, WAVE_EFF, DATA_DIR, ZPCORR_DIR, ZP_TABLE
from .io import print_level

SPLUS_ARGS = {
    'redo': ['r', dict(action='store_true', default=False, help='Enable redo mode to overwrite final cubes.')],
    'clean': ['c', dict(action='store_true', default=False, help='Clean intermediate files after processing.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'bands': ['b', dict(default=list(WAVE_EFF.keys()), nargs='+', help='List of S-PLUS bands (space separated).')],
    'size': ['l', dict(default=500, type=int, help='Size of the cube in pixels.')],
    'angsize': ['a', dict(default=50, type=float, help="Galaxy's Angular size in arcsec.")],
    'work_dir': ['w', dict(default=getcwd(), help='Working directory.')],
    'output_dir': ['o', dict(default=getcwd(), help='Output directory.')],
    'sextractor': ['x', dict(default='sex', help='Path to SExtractor executable.')],
    'class_star': ['p', dict(default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'debug': ['D', dict(action='store_true', default=False, help='Enable debug mode.')],
    'satur_level': ['S', dict(default=1600.0, type=float, help='Saturation level for the png images.')],
    'data_dir': ['d', dict(default=DATA_DIR, help='Data directory. All input data should be located here.')],
    'zpcorr_dir': ['Z', dict(default=ZPCORR_DIR, help='Zero-point correction directory (relative to DATA_DIR).')],
    'zp_table': ['z', dict(default=ZP_TABLE, help='Zero-point table (relative to DATA_DIR).')],
    'back_size': ['B', dict(default=64, type=int, help='Background mesh size for SExtractor.')],
    'detect_thresh': ['T', dict(default=1.1, type=float, help='Detection threshold for SExtractor.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')]
}

class readFileArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(readFileArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

def scubes_parse_arguments(argv):
    # CREATE PARSER
    _formatter = lambda prog: RawDescriptionHelpFormatter(prog, max_help_position=30)
    parser = readFileArgumentParser(fromfile_prefix_chars='@', description=SPLUS_PROG_DESC, formatter_class=_formatter)

    for k, v in SPLUS_ARGS.items():
        long_option = k
        short_option, kwargs = v
        option_string = []
        if short_option != '':
            option_string.append(f'-{short_option}')
        option_string.append(f'--{long_option}')
        if (long_option != 'verbose') and (kwargs.get('default') is not None):
            _tmp = kwargs['help']
            kwargs['help'] = f'{_tmp} Default value is %(default)s'
        parser.add_argument(*option_string, **kwargs)

    # positional arguments
    parser.add_argument('tile', metavar='SPLUS_TILE', help='Name of the S-PLUS tile')
    parser.add_argument('ra', type=str, metavar='RA', help="Galaxy's right ascension")
    parser.add_argument('dec', type=str, metavar='DEC', help="Galaxy's declination")
    parser.add_argument('galaxy', metavar='GALAXY_NAME', help="Galaxy's name")
    parser.add_argument('specz', type=float, metavar='REDSHIFT', help='Spectroscopic or photometric redshift of the galaxy')    

    args = parser.parse_args(args=argv[1:])
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