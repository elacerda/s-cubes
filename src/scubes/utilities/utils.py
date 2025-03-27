import sys
from os.path import isfile
from astropy.io import fits

from .io import print_level
from .args import create_parser

from .. import __author__
from ..entry_points import SPLUS_MOTD_TOP, SPLUS_MOTD_MID, SPLUS_MOTD_BOT, SPLUS_MOTD_SEP

from ..constants import BANDS

#############################################################################
#############################################################################
#############################################################################

GET_LUPTON_RGB_DESC = f'''
{SPLUS_MOTD_TOP} | get_lupton_RGB entry-point script:
{SPLUS_MOTD_MID} | Downloads S-PLUS RGB stamp created
{SPLUS_MOTD_BOT} | using splusdata API.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''

GET_LUPTON_RGB_ARGS = {
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

    parser = create_parser(args_dict=GET_LUPTON_RGB_ARGS, program_description=GET_LUPTON_RGB_DESC)
    args = get_lupton_RGB_argsparse(parser.parse_args(args=sys.argv[1:]))
    conn = connect_splus_cloud(args.username, args.password)
    _get_lupton_RGB(conn, args)

#############################################################################
#############################################################################
#############################################################################

SCUBE_SEX_MASK_STARS_ARGS = {
    # optional arguments
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'no_interact': ['N', dict(action='store_true', default=False, help='Run only the automatic mask (a.k.a. do not check final mask)')],
    
    'sextractor': ['x', dict(default='sex', help='Path to SExtractor executable.')],
    'class_star': ['p', dict(default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation.')],
    'satur_level': ['S', dict(default=1600.0, type=float, help='Saturation level for the png images.')],
    'back_size': ['B', dict(default=64, type=int, help='Background mesh size for SExtractor.')],
    'detect_thresh': ['T', dict(default=1.1, type=float, help='Detection threshold for SExtractor.')],
    'estimate_fwhm': ['F', dict(action='store_true', default=False, help='Runs SExtractor two times estimating the SEEING_FWHM of the detection image.')],

    # positional arguments
    'cube_path': ['pos', dict(metavar='CUBEFITSFILE', help="Galaxy's S-Cube FITS file")],
}

SCUBE_SEX_MASK_STARS_DESC = f'''
{SPLUS_MOTD_TOP} | scube_sex_mask_stars entry-point script:
{SPLUS_MOTD_MID} | Uses S-PLUS detection image and SExtractor 
{SPLUS_MOTD_BOT} | to identify stars on the FOV using the S-Cube
{SPLUS_MOTD_SEP} + FITSFILE of a galaxy as input.

   {__author__}

'''

def scube_sex_mask_stars_argsparse(args):
    '''
    A particular parser of the command-line arguments for `scube_sex_mask_stars_cube` 
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
    from shutil import which
    from .io import convert_coord_to_degrees

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

    for key, value in args.__dict__.items():
        print_level(f'control obj - key: {key} - value: {value}', 2, args.verbose)

    args.header = fits.getheader(args.cube_path, 0)
    h = args.header
    args.ra, args.dec = convert_coord_to_degrees(h['RA'], h['DEC'])
    args.size = int(h['SIZE'])
    args.bands = BANDS
    args.tile = h['TILE']
    args.galaxy = h['GALAXY']

    return args

def scube_sex_mask_stars():
    '''
    Uses S-PLUS detection image and SExtractor to identify stars on the FOV.
    This entry-point script uses a S-Cube as input to gather the needed 
    information to the retrieve the correct detection image stamp.

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

    parser = create_parser(args_dict=SCUBE_SEX_MASK_STARS_ARGS, program_description=SCUBE_SEX_MASK_STARS_DESC)
    args = scube_sex_mask_stars_argsparse(parser.parse_args(args=sys.argv[1:]))
    conn = connect_splus_cloud(args.username, args.password)
    detection_image = f'{args.galaxy}_detection.fits'

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
        #sys.exit(1)

    maskStars(args=args, detection_image=detection_image, lupton_rgb=_get_lupton_RGB(conn, args, save_img=False), output_dir='.')

#############################################################################
#############################################################################
#############################################################################

MLTOHEADER_DESC = f'''
{SPLUS_MOTD_TOP} | ml2header entry-point script:
{SPLUS_MOTD_MID} | Inputs S-CUBES masterlist information
{SPLUS_MOTD_BOT} | to the primary header of a raw cube.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''
MLTOHEADER_ARGS = {
    'force': ['f', dict(action='store_true', help='Force the update the value of existent header keys')],
    'cube': ['pos', dict(metavar='CUBE', help="Path to a Galaxy's S-CUBES fits")], 
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

def ml2header_updheader(cube_filename, ml_table, force=False):
    '''
    Updates a S-CUBES raw cube primary header with the masterlist 
    information.

    Parameters
    ----------
    cube_filename : str
        Path to S-CUBES raw cube.
    
    ml_table : :class:`astropy.table.table.Table`
        Masterlist read using :meth:`astropy.io.ascii.read`
        
    force : bool, optional
        Force the update the key value is the key is existent at the 
        S-CUBES header. 
    '''
    import numpy as np

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
            if v is np.ma.masked:
                v = None
            if '__' in col:
                col, desc = col.split('__')
            if not force and (col in hdu.header):
                continue 
            if col == 'FIELD' or col == 'SNAME':
                continue
            if col == 'SIZE':
                col = 'SIZE_ML'
                desc = 'SIZE masterlist'
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
    
    # update masterlist information
    ml2header_updheader(args.cube, args.ml, args.force)

#############################################################################
#############################################################################
#############################################################################

SPLOTS_DESC = f'''
{SPLUS_MOTD_TOP} | scube_plots entry-point script:
{SPLUS_MOTD_MID} | Creates a set of images from a 
{SPLUS_MOTD_BOT} | SCUBE fits file.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''
SPLOTS_ARGS = {
    'show': ['S', dict(action='store_true', default=False, help='Show plots during execution')],
    'cube': ['pos', dict(metavar='CUBE', help="Path to a Galaxy's S-CUBES fits.")], 
}

def splots():
    '''
    Entry-point function to generate plots from a SCUBE.

    Returns
    -------
    None
    '''
    from .plots import scube_plots

    parser = create_parser(args_dict=SPLOTS_ARGS, program_description=SPLOTS_DESC)
    args = parser.parse_args(args=sys.argv[1:])

    splots = scube_plots(filename=args.cube, block=args.show)

    #######################################
    ################# IMGs ################
    #######################################
    ofile = f'{splots.scube.galaxy}_imgs_3Dflux.png'
    splots.images_3D_plot(output_filename=None, FOV=140)

    ofile = f'{splots.scube.galaxy}_imgs_mag.png'
    splots.images_mag_plot(output_filename=ofile, cmap='magma_r')

    ofile = f'{splots.scube.galaxy}_imgs_emag.png'
    splots.images_emag_plot(output_filename=ofile, cmap='magma_r')

    ofile = f'{splots.scube.galaxy}_imgs_flux.png'
    splots.images_flux_plot(output_filename=ofile, cmap='magma')

    ofile = f'{splots.scube.galaxy}_imgs_SN.png'
    splots.images_SN_plot(output_filename=ofile, cmap='magma')

    #######################################
    ################# SKY #################
    #######################################
    sky, _ = splots.scube.get_iso_sky(
        ref_mag_filt='rSDSS',
        isophotal_limit=25, 
        isophotal_medsize=30, 
        stars_mask=None, 
        n_sigma=3, 
        n_iter=5,
        clip_neg=True,
    )
    il = sky['isophotal_limit']
    im = sky['isophotal_medsize']
    ofile = f'{splots.scube.galaxy}_sky_spec_iso{il}med{im}.png'
    splots.sky_spec_plot(sky, output_filename=ofile)

    #######################################
    ################# RGBs ################
    #######################################
    rgblist = [
        [9, 7, 5],
        [8, 7, 5],
        [8, 9, 0],
        [11, 5, 0],
        [9, [3, 4, 5], [0, 1, 2]],
        [8, 9, 5],
        [8, 5, [0, 1, 2, 3, 4]]
    ] 
    titlelist = [
        '(i, r, g)',
        '(J0660, r, g)',
        '(J0660, i, u)',
        '(z, g, u)',
        '(i, J0410+J0430+g, u+J0378+J0395)',
        '(J0660, i, g)',
        '(J0660, g, u+J0378+J0395+J0410+J0430)'
    ]
    for rgb, title, in zip(rgblist, titlelist):
        s = [ str(x).replace(',','').replace(' ', '').replace('[','').replace(']','') for x in rgb]
        kw = dict(
            rgb=rgb,
            rgb_f=[1, 1, 1],
            pminmax=[1.5, 98.5],
            Q=3,
            stretch=130,
            im_max=180,
            minimum=(15, 15, 15),
            title=title       
        )
        splots.LRGB_plot(output_filename=f'{splots.scube.galaxy}_RGB_{s[0]}-{s[1]}-{s[2]}.png', **kw)

    ofile = f'{splots.scube.galaxy}_LRGB_centspec.png'
    splots.LRGB_centspec_plot(output_filename=ofile)

    #######################################
    ################# SPEC ################
    #######################################
    ofile = f'{splots.scube.galaxy}_rings_spec.png'
    #pa = 1 - splots.scube.primary_header['HIERARCH ELLIPTICITY']
    #ba = splots.scube.primary_header['B_IMAGE']/splots.scube.primary_header['A_IMAGE']
    #theta = splots.scube.primary_header['HIERARCH THETA_IMAGE']
    pa = 0
    ba = 1
    theta = None
    splots.rings_spec_plot(
        output_filename=ofile, 
        pa=pa,
        ba=ba,
        theta=theta,
        rad_scale=1,
        mode='mean',
        sky_mask=sky['mask__yx'],
        rad_mask=None,
    )

    '''
    ofile = f'{splots.scube.galaxy}_imgs_eflux.png'
    splots.images_eflux_plot(output_filename=ofile, cmap='magma')

    ofile = f'{splots.scube.galaxy}_SN_filters.png'
    splots.SN_filters_plot(output_filename=ofile, SN_range=[0, 10], valid_mask__yx=None, bins=50)

    contour_levels = [21, 23, 24]
    ofile = f'{splots.scube.galaxy}_contours_mag_21_23_24.png'
    splots.contour_plot(output_filename=ofile, contour_levels=contour_levels)

    ofile = f'{splots.scube.galaxy}_intarea_rad50pix_spec.png'
    splots.int_area_spec_plot(output_filename=ofile, pa_deg=0, ba=1, R_pix=50)    
    '''

SCUBE_MASK_DESC = f'''
{SPLUS_MOTD_TOP} | scube_mask entry-point script: Masks stars, 
{SPLUS_MOTD_MID} | evaluate sky mean flux and creates the galaxy SCUBE 
{SPLUS_MOTD_BOT} | fits file with the estimated galaxy size and metainfo.
{SPLUS_MOTD_SEP} + 

   {__author__}

'''

def sky_or_int(arg):
    import argparse as ap
    try:
        return float(arg)  # try convert to float
    except ValueError:
        pass
    if arg == 'sky':
        return arg
    raise ap.ArgumentTypeError("x must be a number or 'sky'")

help_star_calc = '''
    Configures how the star size is calculated when using --star_fwhm_individual_calc option.
    'sky' -> the size is defined when flux == sky
    Number -> favtor the multiply the FWHM to get the size of each star.
'''
help_isomedsize = 'Size of the window used for the median filter when smoothing the mask'
help_isolimit = 'The threshold value for the reference magnitude used to mask'
help_star_fwhm = '''
    If True calculates the FWHM used to estimate the star radius individually.
'''

SCUBE_MASK_ARGS = {
    'cube': ['pos', dict(metavar='SCUBE', help="Path to a Galaxy's S-CUBES fits.")],
    'extra_pix': ['P', dict(default=5, type=int, help='Extra pixels to be added to the final masked SCUBE.')],
    'mask_stars_threshold': ['m', dict(default=20, type=int, help='Minimum magnitude threshold. The script searches local density maxima that have a peak amplitude greater than threshold')],
    'mask_stars_bands': ['b', dict(default=7, nargs='+', type=int, help='List of S-PLUS bands (space separated) to create the detection image.')],
    'xsource_std_f': ['X', dict(default=2, type=int, help='Factor to be multiplied by the threshold for the extra sources detection.')],
    'star_size_calc': ['S', dict(default=2, type=sky_or_int, help=help_star_calc)],
    'star_fwhm_individual_calc': ['F', dict(default=False, action='store_true', help=help_star_fwhm)],
    'mask_isophotal_medsize': ['', dict(default=30, type=int, help=help_isomedsize + '.')],
    'mask_isophotal_limit': ['', dict(default=24, type=int, help=help_isolimit)],
    'sky_isophotal_medsize': ['', dict(default=30, type=int, help=help_isomedsize + ' of the sky.')],
    'sky_isophotal_limit': ['', dict(default=25, type=int, help=help_isolimit + ' the sky.')],
    'sky_n_sigma': ['', dict(default=3, type=int, help='The threshold number of standard deviations to use for clipping outliers in sky flux values.')],
    'sky_n_iter': ['', dict(default=5, type=int, help='The number of iterations to perform for sigma clipping to remove outliers.')],
    'sky_clip_neg': ['', dict(default=False, action='store_true', help='If True, clip both negative and positive outliers. If False, only clip positive outliers.')],
    'no_interact': ['N', dict(action='store_true', default=False, help='Run automatic stars mask.')],
}

def scube_mask():
    import matplotlib.pyplot as plt
    
    from .masks import masks_builder
    from .readscube import read_scube

    parser = create_parser(args_dict=SCUBE_MASK_ARGS, program_description=SCUBE_MASK_DESC)
    args = parser.parse_args(args=sys.argv[1:])

    scube = read_scube(args.cube)
    masks_builder(scube, args).mask_procedure()