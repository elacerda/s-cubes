from os import getcwd
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from .constants import PROG_DESC, WAVE_EFF, DATA_DIR, ZPCORR_DIR, ZP_TABLE

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

def create_parser():
    """
        Parse command line arguments for the program.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    _formatter = lambda prog: RawDescriptionHelpFormatter(prog, max_help_position=30)
    parser = readFileArgumentParser(fromfile_prefix_chars='@', description=PROG_DESC, formatter_class=_formatter)
    
    parser.add_argument('-r', '--redo', action='store_true', default=False, help='Enable redo mode to overwrite final cubes')
    parser.add_argument('-c', '--clean', action='store_true', default=False, help='Clean intermediate files after processing')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='Force overwrite of existing files')
    
    # parser.add_argument('-s', '--savestamps', action='store_false', default=True, help='Save stamps')

    parser.add_argument('-b', '--bands', default=list(WAVE_EFF.keys()), help='List of S-PLUS bands')
    parser.add_argument('-l', '--size', default=500, type=int, help='Size of the cube in pixels')
    parser.add_argument('-a', '--angsize', default=50, type=float, help="Galaxy's Angular size in arcsec")
    parser.add_argument('-w', '--work_dir', default=getcwd(), help='Working directory')
    parser.add_argument('-o', '--output_dir', default=getcwd(), help='Output directory')
    parser.add_argument('-x', '--sextractor', default='sex', help='Path to SExtractor executable')
    parser.add_argument('-p', '--class_star', default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    # parser.add_argument('-q', '--tile_dir', default=None, help='Directory where the S-PLUS images are stored.\n' 'Default is work_dir/tile')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--satur_level', default=1600.0, type=float, help='Saturation level for the png images. Default is 1600.0')
    parser.add_argument('--data_dir', default=DATA_DIR, help='Data directory')
    parser.add_argument('--zpcorr_dir', default=ZPCORR_DIR, help='Zero-point correction directory')
    parser.add_argument('--zp_table', default=ZP_TABLE, help='Zero-point table')
    parser.add_argument('--back_size', default=64, type=int, help='Background mesh size for SExtractor. Default is 64')
    parser.add_argument('--detect_thresh', default=1.1, type=float, help='Detection threshold for SExtractor. Default is 1.1')

    # positional arguments
    parser.add_argument('tile', metavar='SPLUS_TILE', help='Name of the S-PLUS tile')
    parser.add_argument('ra', type=str, metavar='RA', help="Galaxy's right ascension")
    parser.add_argument('dec', type=str, metavar='DEC', help="Galaxy's declination")
    parser.add_argument('galaxy', metavar='GALAXY_NAME', help="Galaxy's name")
    parser.add_argument('specz', type=float, metavar='REDSHIFT', help='Spectroscopic or photometric redshift of the galaxy')
    
    return parser