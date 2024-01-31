import sys
from os.path import isfile
from astropy.io import fits

from .io import print_level
from .args import create_parser

from .. import __author__
from ..entry_points import SPLUS_MOTD_TOP, SPLUS_MOTD_MID, SPLUS_MOTD_BOT, SPLUS_MOTD_SEP, scubes_argparse

SPLUS_RGB_DESC = f'''
{SPLUS_MOTD_TOP} | get_lupton_RGB entry-point script:
{SPLUS_MOTD_MID} | Downloads S-PLUS RGB stamp created
{SPLUS_MOTD_BOT} | using splusdata API.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''

SPLUS_RGB_ARGS = {
    # optional arguments
    'size': ['l', dict(default=500, type=int, help='Size of the cube in pixels.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'galaxy': ['g', dict(default=None, metavar='GALAXY_NAME', help="Galaxy's name")],

    # positional arguments
    'tile': ['pos', dict(metavar='SPLUS_TILE', help='Name of the S-PLUS tile')],
    'ra': ['pos', dict(metavar='RA', help="Galaxy's right ascension")],
    'dec': ['pos', dict(metavar='DEC', help="Galaxy's declination")],
}

def get_lupton_RGB_argsparse(args):
    '''
    A particular parser of the command-line arguments for `get_lupton_RGB`
    entry-point script.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :meth:`argparse.ArgumentParser.parse_args`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
    from .io import convert_coord_to_degrees

    for key, value in args.__dict__.items():
        print_level(f'control obj - key: {key} - value: {value}', 2, args.verbose)

    args.ra, args.dec = convert_coord_to_degrees(args.ra, args.dec)

    return args

def _get_lupton_RGB(conn, args, save_img=True):
    '''
    Downloads S-PLUS RGB images using the `splusdata` module.

    Parameters
    ----------
    conn : object
        Connection object to the S-PLUS Cloud.

    args : :class:`argparse.Namespace`
        Parsed command-line arguments.

    save_img : bool, optional
        Flag to save the downloaded image (default is True).

    Returns
    -------
    :class:`PIL.Image.Image`
        Downloaded RGB image.
    '''
    from .splusdata import get_lupton_rgb
 
    if args.galaxy is None:
        args.galaxy = 'OBJECT'
    print_level(f'{args.galaxy} @ {args.tile} - downloading RGB image', 1, args.verbose)

    kw = dict(ra=args.ra, dec=args.dec, size=args.size, option=args.tile)
    fname = f'{args.galaxy}_{args.tile}_{args.size}x{args.size}.png'
    img = get_lupton_rgb(conn, transpose=True, save_img=save_img, filename=fname, **kw)

    return img

def get_lupton_RGB():
    '''
    Entry-point function for downloading S-PLUS RGB images.

    Returns
    -------
    None
    '''
    from .splusdata import connect_splus_cloud

    parser = create_parser(args_dict=SPLUS_RGB_ARGS, program_description=SPLUS_RGB_DESC)
    args = get_lupton_RGB_argsparse(parser.parse_args(args=sys.argv[1:]))
    conn = connect_splus_cloud(args.username, args.password)
    _get_lupton_RGB(conn, args)

SPLUS_SEX_MASK_STARS_ARGS = {
    # optional arguments
    'size': ['l', dict(default=500, type=int, help='Size of the cube in pixels.')],
    'sextractor': ['x', dict(default='sex', help='Path to SExtractor executable.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'class_star': ['p', dict(default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'no_interact': ['N', dict(action='store_true', default=False, help='Run only the automatic mask (a.k.a. do not check final mask)')],
    'satur_level': ['S', dict(default=1600.0, type=float, help='Saturation level for the png images.')],
    'back_size': ['B', dict(default=64, type=int, help='Background mesh size for SExtractor.')],
    'detect_thresh': ['T', dict(default=1.1, type=float, help='Detection threshold for SExtractor.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'estimate_fwhm': ['F', dict(action='store_true', default=False, help='Runs SExtractor two times estimating the SEEING_FWHM of the detection image.')],
    'bands': ['b', dict(default='G,R,I,Z', help='List of S-PLUS bands.')],
    'galaxy': ['g', dict(default=None, metavar='GALAXY_NAME', help="Galaxy's name")],

    # positional arguments
    'tile': ['pos', dict(metavar='SPLUS_TILE', help='Name of the S-PLUS tile')],
    'ra': ['pos', dict(metavar='RA', help="Galaxy's right ascension")],
    'dec': ['pos', dict(metavar='DEC', help="Galaxy's declination")],
}

SPLUS_SEX_MASK_STARS_DESC = f'''
{SPLUS_MOTD_TOP} | sex_mask_stars entry-point script:
{SPLUS_MOTD_MID} | Uses S-PLUS detection image and SExtractor 
{SPLUS_MOTD_BOT} | to identify stars on the FOV. 
{SPLUS_MOTD_SEP} + 

   {__author__}

'''

def sex_mask_stars_args(args):
    '''
    A specific parser for command-line arguments used in the `sex_mask_stars` function.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :meth:`argparse.ArgumentParser.parse_args`

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed command-line arguments.
    '''
    return get_lupton_RGB_argsparse(scubes_argparse(args))
    
def sex_mask_stars():
    '''
    Uses S-PLUS detection image and SExtractor to identify stars on the FOV.

    Raises
    ------
    SystemExit
        If the detection file exists.

    Returns
    -------
    None
    '''
    from ..mask_stars import maskStars
    from ..headers import get_author
    
    from .splusdata import connect_splus_cloud, detection_image_hdul

    parser = create_parser(args_dict=SPLUS_SEX_MASK_STARS_ARGS, program_description=SPLUS_SEX_MASK_STARS_DESC)
    args = sex_mask_stars_args(parser.parse_args(args=sys.argv[1:]))
    conn = connect_splus_cloud(args.username, args.password)
    if args.galaxy is None:
        args.galaxy = 'OBJECT'
    prefix_filename = f'{args.galaxy}_{args.tile}_{args.size}x{args.size}'
    detection_image = f'{prefix_filename}_detection.fits'

    if not isfile(detection_image) or args.force:
        print_level(f'{args.galaxy} @ {args.tile} - downloading detection image')
        kw = dict(ra=args.ra, dec=args.dec, size=args.size, bands=args.bands, option=args.tile)
        hdul = detection_image_hdul(conn, **kw)

        author = get_author(hdul[1].header)

        # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
        if author == 'unknown':
            author = 'scubes'
            hdul[1].header.set('AUTHOR', value=author, comment='Who ran the software')

        # SAVE DETECTION FITS
        hdul.writeto(detection_image, overwrite=args.force)
    else:
        print_level('Detection file exists.')
        sys.exit(1)

    maskStars(args=args, detection_image=detection_image, lupton_rgb=_get_lupton_RGB(conn, args, save_img=False), output_dir='.')

MLTOHEADER_DESC = f'''
{SPLUS_MOTD_TOP} | ml2header entry-point script:
{SPLUS_MOTD_MID} | Inputs S-CUBES masterlist information
{SPLUS_MOTD_BOT} | to the primary header of a raw cube.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''
MLTOHEADER_ARGS = {
    'cube': ['pos', dict(metavar='CUBE', help="Path to a Galaxy's S-CUBES fits.")], 
    'masterlist': ['pos', dict(metavar='MASTERLIST', help='Path to masterlist file')]
}

def ml2header_argparse(args):
    '''
    A particular parser of the command-line arguments for `ml2header` entry-point script.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :meth:`argparse.ArgumentParser.parse_args`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
    from astropy.io import ascii

    try:
        _hdul = fits.open(args.cube)
    except:
        print_level(f'{args.cube}: unable to read file')
        sys.exit(1)
    _hdul.close()

    try:
        args.ml = ascii.read(args.masterlist)
    except:
        print_level(f'{args.masterlist}: unable to read file')
        sys.exit(1)

    return args

def ml2header_updheader(cube_filename, ml_table):
    '''
    Updates a S-CUBES raw cube primary header with the masterlist 
    information.

    Parameters
    ----------
    cube_filename : str
        Path to S-CUBES raw cube.
    
    ml_table : :class:`astropy.table.table.Table`
        Masterlist read using :meth:`astropy.io.ascii.read`.
    '''
    with fits.open(cube_filename, 'update') as hdul:
        hdu = hdul['PRIMARY']

        # SNAME CONTROL
        sname = hdu.header.get('GALAXY', None)
        if sname is None:
            print_level('header: missing SNAME information')
            sys.exit(1)
        if sname not in ml_table['SNAME']:
            print_level(f'masterlist: {sname}: missing SNAME information')
            sys.exit(1)

        mlcut = ml_table[ml_table['SNAME'] == sname]
        for col in ml_table.colnames:
            v = mlcut[col][0]
            desc = None
            if '__' in col:
                col, _, desc = col.split('_')
            hdu.header.set(col, value=v, comment=desc)


def ml2header():
    '''
    Entry-point function to add the masterlist information to the primary
    header of a S-CUBES original cube.

    Raises
    ------
    SystemExit
        If some input paths are not existent or present some problem or if
        the primary header of the cube FITS miss some needed information, 
        like the S-CUBES name information.

    Returns
    -------
    None
    '''
    parser = create_parser(args_dict=MLTOHEADER_ARGS, program_description=MLTOHEADER_DESC)
    args = ml2header_argparse(parser.parse_args(args=sys.argv[1:]))

