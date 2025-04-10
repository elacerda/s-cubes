import sys
from os import getcwd
from shutil import which

from . import __author__, __version__, __filters_table__  #, __zp_cat__, __zpcorr_path__

from .constants import BANDS

from .utilities.io import print_level
from .utilities.args import create_parser

SPLUS_MOTD_TOP = '┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐ '
SPLUS_MOTD_MID = '└─┐───│  │ │├┴┐├┤ └─┐ '
SPLUS_MOTD_BOT = '└─┘   └─┘└─┘└─┘└─┘└─┘ '
SPLUS_MOTD_SEP = '----------------------'

SCUBES_PROG_DESC = f'''
{SPLUS_MOTD_TOP} | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES. 
{SPLUS_MOTD_MID} | S-CUBES is an organized FITS file with data, errors, 
{SPLUS_MOTD_BOT} | mask and metadata about some galaxy present on any 
{SPLUS_MOTD_SEP} + S-PLUS observed tile. Any problem contact:

   {__author__}

The input values of RA and DEC will be converted to degrees using the 
scubes.utilities.io.convert_coord_to_degrees(). All scripts with RA 
and DEC inputs parse angles in two different units:

- **hourangle**: using *hms* divisors; Ex: *10h37m2.5s*
- **degrees**: using *:* or *dms*  divisors; Ex: *10:37:2.5* or *10d37m2.5s*

Note that *10h37m2.5s* is a totally different angle from *10:37:2.5* 
(*159.26 deg* and *10.62 deg* respectively).

'''

SCUBES_ARGS = {
    # optional arguments
    'redo': ['r', dict(action='store_true', default=False, help='Enable redo mode to overwrite final cubes.')],
    'clean': ['c', dict(action='store_true', default=False, help='Clean intermediate files after processing.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'bands': ['b', dict(default=BANDS, nargs='+', help='List of S-PLUS bands (space separated).')],
    'size': ['l', dict(default=500, type=int, help='Size of the cube in pixels. If size is a odd number, the program will choose the closest even integer.')],
    'work_dir': ['w', dict(default=getcwd(), help='Working directory.')],
    'output_dir': ['o', dict(default=getcwd(), help='Output directory.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'debug': ['D', dict(action='store_true', default=False, help='Enable debug mode.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'det_img': ['I', dict(action='store_true', default=False, help='Downloads detection image for the stamp. Needed if --mask_stars is active.')],
    'remove_downloaded_data': ['R', dict(action='store_true', default=False, help='Remove the downloaded data from splusdata at the end of the run.')],
    'data_release': ['d', dict(default='dr4', type=str, help='Select S-PLUS Data Release')],

    #'zpcorr_dir': ['Z', dict(default=__zpcorr_path__, help='Zero-point correction directory.')],
    #'zp_table': ['z', dict(default=__zp_cat__, help='Zero-point table.')],
    #'no_interact': ['N', dict(action='store_true', default=False, help='Run only the automatic stars mask (a.k.a. do not check final mask)')],
    #'mask_stars': ['M', dict(action='store_true', default=False, help='Run SExtractor to auto-identify stars on stamp.')],
    #'sextractor': ['x', dict(default='sex', help='Path to SExtractor executable.')],
    #'class_star': ['p', dict(default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation.')],
    #'satur_level': ['S', dict(default=1600.0, type=float, help='Saturation level for the png images.')],
    #'back_size': ['B', dict(default=64, type=int, help='Background mesh size for SExtractor.')],
    #'detect_thresh': ['T', dict(default=1.1, type=float, help='Detection threshold for SExtractor.')],
    #'estimate_fwhm': ['F', dict(action='store_true', default=False, help='Runs SExtractor two times estimating the SEEING_FWHM of the detection image.')],

    # positional arguments
    'tile': ['pos', dict(metavar='SPLUS_TILE', help='Name of the S-PLUS tile')],
    'ra': ['pos', dict(metavar='RA', help="Galaxy's right ascension")],
    'dec': ['pos', dict(metavar='DEC', help="Galaxy's declination")],
    'galaxy': ['pos', dict(metavar='GALAXY_NAME', help="Galaxy's name")],
}

def scubes_argparse(args):
    '''
    A particular parser of the command-line arguments for `scubes` entry-point script.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :meth:`argparse.ArgumentParser.parse_args`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
    # closest even
    min_size = args
    args.size = round(args.size/2)*2
    
    '''
    if args.mask_stars:
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
    '''
    return args

def scubes():
    '''
    Entry-point function for creating S-PLUS galaxy data cubes (S-CUBES).

    Raises
    ------
    SystemExit
        If SExtractor is not found.

    Returns
    -------
    None
    '''
    from .core import SCubes

    parser = create_parser(args_dict=SCUBES_ARGS, program_description=SCUBES_PROG_DESC)
    # ADD VERSION OPTION
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    args = scubes_argparse(parser.parse_args(args=sys.argv[1:]))
    scubes = SCubes(args)
    scubes.create_cube(flam_scale=None)

SCUBESML_PROG_DESC = f'''
{SPLUS_MOTD_TOP} | scubesml entry-point script:
{SPLUS_MOTD_MID} | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES
{SPLUS_MOTD_BOT} | using the masterlist information as input.
{SPLUS_MOTD_SEP} + 

   {__author__}
'''

SCUBESML_ARGS = {
    # optional arguments
    'redo': ['r', dict(action='store_true', default=False, help='Enable redo mode to overwrite final cubes.')],
    'clean': ['c', dict(action='store_true', default=False, help='Clean intermediate files after processing.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'bands': ['b', dict(default=BANDS, nargs='+', help='List of S-PLUS bands (space separated).')],
    'size_multiplicator': ['S', dict(default=10, type=float, help='Factor to multiply the SIZE__pix value of the masterlist to create the galaxy size. If size is a odd number, the program will choose the closest even integer.')],
    'min_size': ['m', dict(default=200, type=int, help='Minimal size of the cube in pixels. If size negative, uses the original value of size calculation.')],
    'data_release': ['d', dict(default='dr4', type=str, help='Select S-PLUS Data Release')],
    'work_dir': ['w', dict(default=getcwd(), help='Working directory.')],
    'output_dir': ['o', dict(default=getcwd(), help='Output directory.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'debug': ['D', dict(action='store_true', default=False, help='Enable debug mode.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'remove_downloaded_data': ['R', dict(action='store_true', default=False, help='Remove the downloaded data from splusdata at the end of the run.')],

    #'zpcorr_dir': ['Z', dict(default=__zpcorr_path__, help='Zero-point correction directory.')],
    #'zp_table': ['z', dict(default=__zp_cat__, help='Zero-point table.')],

    # positional arguments
    #'tile': ['pos', dict(metavar='SPLUS_TILE', help='Name of the S-PLUS tile')],
    #'ra': ['pos', dict(metavar='RA', help="Galaxy's right ascension")],
    #'dec': ['pos', dict(metavar='DEC', help="Galaxy's declination")],
    'galaxy': ['pos', dict(metavar='GALAXY_SNAME', help="Galaxy's masterlist nickname")],
    'masterlist': ['pos', dict(metavar='MASTERLIST', help='Path to masterlist file')]
}

def scubesml_argparse(args):
    '''
    A particular parser of the command-line arguments for `scubesml` entry-point script.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :meth:`argparse.ArgumentParser.parse_args`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
    # closest even
    from astropy.io import ascii

    args.mask_stars = False
    args.det_img = False

    try:
        args.ml = ascii.read(args.masterlist)
    except:
        print_level(f'{args.masterlist}: unable to read file')
        sys.exit(1)
    
    # Retrieve scubes positional arguments
    mlcut = args.ml[args.ml['SNAME'] == args.galaxy]
    args.tile = mlcut['FIELD'][0]
    args.ra = mlcut['RA__deg'][0]
    args.dec = mlcut['DEC__deg'][0]
    min_size = args.min_size
    args.size = max(round(args.size_multiplicator*float(mlcut['SIZE__pix'])/2)*2, min_size)

    return args

def scubesml():
    '''
    Entry-point function for creating S-PLUS galaxy data cubes (S-CUBES)
    using the masterlist for the input arguments.

    Raises
    ------
    SystemExit
        If masterlist not found

    Returns
    -------
    None
    '''

    from .core import SCubes
    from .utilities.utils import ml2header_updheader

    parser = create_parser(args_dict=SCUBESML_ARGS, program_description=SCUBESML_PROG_DESC)
    # ADD VERSION OPTION
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    args = scubesml_argparse(parser.parse_args(args=sys.argv[1:]))
    scubes = SCubes(args)
    scubes.create_cube(flam_scale=None)

    # update masterlist information
    ml2header_updheader(scubes.cube_path, args.ml)

SCUBES_FILTERS_PROG_DESC = f'''
{SPLUS_MOTD_TOP} | scubes_filters entry-point script:
{SPLUS_MOTD_MID} | 
{SPLUS_MOTD_BOT} | Print the S-PLUS filters information.
{SPLUS_MOTD_SEP} + 

   {__author__}
'''

SCUBES_FILTERS_ARGS = {
    # optional arguments
    'decimals': ['d', dict(default=99, type=int, help='Format the numbers at the table.')],
}

def scubes_filters():
    '''
    Print the S-PLUS filters information.

    Raises
    ------
    SystemExit
        If masterlist not found

    Returns
    -------
    None
    '''

    from . import __filters_table__

    parser = create_parser(args_dict=SCUBES_FILTERS_ARGS, program_description=SCUBES_FILTERS_PROG_DESC)
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    args = parser.parse_args(args=sys.argv[1:])

    __filters_table__.round(decimals=args.decimals)

    __filters_table__.pprint_all()