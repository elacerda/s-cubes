import sys
from shutil import which
from os.path import isfile
from astropy.io import fits
from os import getcwd, remove

from . import __author__, __zp_cat__, __zpcorr_path__
from .constants import WAVE_EFF

from .utilities.io import print_level
from .utilities.args import create_parser

SPLUS_MOTD_TOP = '┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐ '
SPLUS_MOTD_MID = '└─┐───│  │ │├┴┐├┤ └─┐ '
SPLUS_MOTD_BOT = '└─┘   └─┘└─┘└─┘└─┘└─┘ '
SPLUS_MOTD_SEP = '----------------------'

SPLUS_PROG_DESC = f'''
{SPLUS_MOTD_TOP} | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES. 
{SPLUS_MOTD_MID} | S-CUBES is an organized FITS file with data, errors, 
{SPLUS_MOTD_BOT} | mask and metadata about some galaxy present on any 
{SPLUS_MOTD_SEP} + S-PLUS observed tile. Any problem contact:

   {__author__}

'''

SPLUS_ARGS = {
    # optional arguments
    'redo': ['r', dict(action='store_true', default=False, help='Enable redo mode to overwrite final cubes.')],
    'clean': ['c', dict(action='store_true', default=False, help='Clean intermediate files after processing.')],
    'force': ['f', dict(action='store_true', default=False, help='Force overwrite of existing files.')],
    'bands': ['b', dict(default=list(WAVE_EFF.keys()), nargs='+', help='List of S-PLUS bands (space separated).')],
    'size': ['l', dict(default=500, type=int, help='Size of the cube in pixels.')],
    #'angsize': ['a', dict(default=50, type=float, help="Galaxy's Angular size in arcsec.")],
    'work_dir': ['w', dict(default=getcwd(), help='Working directory.')],
    'output_dir': ['o', dict(default=getcwd(), help='Output directory.')],
    'sextractor': ['x', dict(default='sex', help='Path to SExtractor executable.')],
    'class_star': ['p', dict(default=0.25, type=float, help='SExtractor CLASS_STAR parameter for star/galaxy separation.')],
    'verbose': ['v', dict(action='count', default=0, help='Verbosity level.')],
    'debug': ['D', dict(action='store_true', default=False, help='Enable debug mode.')],
    'satur_level': ['S', dict(default=1600.0, type=float, help='Saturation level for the png images.')],
    'zpcorr_dir': ['Z', dict(default=__zpcorr_path__, help='Zero-point correction directory.')],
    'zp_table': ['z', dict(default=__zp_cat__, help='Zero-point table.')],
    'back_size': ['B', dict(default=64, type=int, help='Background mesh size for SExtractor.')],
    'detect_thresh': ['T', dict(default=1.1, type=float, help='Detection threshold for SExtractor.')],
    'username': ['U', dict(default=None, help='S-PLUS Cloud username.')],
    'password': ['P', dict(default=None, help='S-PLUS Cloud password.')],
    'mask_stars': ['M', dict(action='store_true', default=False, help='Run SExtractor to auto-identify stars on stamp.')],
    'det_img': ['I', dict(action='store_true', default=False, help='Downloads detection image for the stamp. Needed if --mask_stars is active.')],
    'estimate_fwhm': ['F', dict(action='store_true', default=False, help='Runs SExtractor two times estimating the SEEING_FWHM of the detection image.')],

    # positional arguments
    'tile': ['pos', dict(metavar='SPLUS_TILE', help='Name of the S-PLUS tile')],
    'ra': ['pos', dict(metavar='RA', help="Galaxy's right ascension")],
    'dec': ['pos', dict(metavar='DEC', help="Galaxy's declination")],
    'galaxy': ['pos', dict(metavar='GALAXY_NAME', help="Galaxy's name")],
    'specz': ['pos', dict(type=float, metavar='REDSHIFT', help='Spectroscopic or photometric redshift of the galaxy')],
}

def scubes_argparse(args):
    '''
    A particular parser of the command-line arguments for `scubes` entry-point script.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :function:`argparse.ArgumentParser.parse_args()`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
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

SPLUS_RGB_DESC = f'''
{SPLUS_MOTD_TOP} | Downloads S-PLUS RGB Image.
{SPLUS_MOTD_MID} | It will download a png file containing the
{SPLUS_MOTD_BOT} | RGB image for a stamp created using splusdata
{SPLUS_MOTD_SEP} + API.

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
        Command-line arguments parsed by :function:`argparse.ArgumentParser.parse_args()`

    Returns
    -------
    :class:`argparse.Namespace`
        Command-line arguments parsed.
    '''
    from .utilities.io import convert_coord_to_degrees

    for key, value in args.__dict__.items():
        print_level(f'control obj - key: {key} - value: {value}', 2, args.verbose)

    args.ra, args.dec = convert_coord_to_degrees(args.ra, args.dec)

    return args

def _get_lupton_RGB(conn, args, save_img=True):
    from .utilities.splusdata import get_lupton_rgb
 
    if args.galaxy is None:
        args.galaxy = 'OBJECT'
    print_level(f'{args.galaxy} @ {args.tile} - downloading RGB image', 1, args.verbose)

    kw = dict(ra=args.ra, dec=args.dec, size=args.size, option=args.tile)
    fname = f'{args.galaxy}_{args.tile}_{args.size}x{args.size}.png'
    img = get_lupton_rgb(conn, transpose=True, save_img=save_img, filename=fname, **kw)

    return img

def get_lupton_RGB():
    from .utilities.splusdata import connect_splus_cloud

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
{SPLUS_MOTD_TOP} | Uses S-PLUS detection image and SExtractor
{SPLUS_MOTD_MID} | to identify stars on the FOV. It will download
{SPLUS_MOTD_BOT} | a FITS file containing the detection image 
{SPLUS_MOTD_SEP} + stamp created using splusdata API.

   {__author__}

'''

def sex_mask_stars_args(args):
    return get_lupton_RGB_argsparse(scubes_argparse(args))

def sex_mask_stars_calc_masks(detection_image, sex_path, input_sex_config, output_sex_parameters, 
                              lupton_rgb, size, class_star=0.2, prefix_filename=None, save_fig=False, 
                              run_DAOfinder=False, unmask_stars=None, fig=None, estimate_fwhm=False, 
                              verbose=0):
    from .utilities.plots import plot_mask
    from .utilities.daofinder import DAOregions
    from .utilities.stats import robustStat
    from .utilities.splusdata import get_lupton_rgb
    from .utilities.sextractor import unmask_sewregions,SEWregions, run_sex
    
    print_level('Calculating mask...')
    print_level('Running SExtractor to get photometry...')

    prefix_filename = 'OBJECT' if prefix_filename is None else prefix_filename
    i = 0 if estimate_fwhm else 1
    while i < 2:
        sewcat = run_sex(
            sex_path=sex_path, 
            detection_fits=detection_image, 
            input_config=input_sex_config, 
            output_params=output_sex_parameters, 
            work_dir=None, 
            output_file=detection_image.replace('detection', 'sexcat'), 
            verbose=verbose
        )
        if not i and estimate_fwhm:
            stats = robustStat(sewcat['table']['FWHM_IMAGE']) 
            psffwhm = stats['median']*0.55
            fits.setval(detection_image, 'HIERARCH OAJ PRO FWHMMEAN', value=psffwhm, comment='', ext=1)
            files_to_remove = ['params.txt', 'conv.txt', 'config.txt', 'default.psf']
            for _f in files_to_remove:
                remove(_f)
            input_sex_config['SEEING_FWHM'] = psffwhm
        i += 1
    
    h = fits.getheader(detection_image, ext=1)
    
    sewregions =  SEWregions(sewcat=sewcat, class_star=class_star, shape=(h.get('NAXIS2'), h.get('NAXIS1')), verbose=verbose)
    
    data = fits.getdata(detection_image, ext=1)
    
    daoregions = None
    if run_DAOfinder:
        daoregions = DAOregions(data=data)
    
    masked_ddata, resulting_mask = unmask_sewregions(data=data, sewregions=sewregions, size=size, unmask_stars=unmask_stars, verbose=verbose)

    fig = plot_mask(
        detection_image=detection_image,
        lupton_rgb=lupton_rgb, 
        masked_ddata=masked_ddata, 
        resulting_mask=resulting_mask, 
        sewregions=sewregions, 
        daoregions=daoregions, 
        save_fig=save_fig,
        prefix_filename=prefix_filename,
        fig=fig,
    )
    return resulting_mask, fig

def sex_mask_stars_loop_mask(sex_path, detection_image, lupton_rgb, input_config, output_parameters, prefix_filename, size, class_star, estimate_fwhm=False, verbose=0):
    from matplotlib import pyplot as plt

    resulting_mask, fig = sex_mask_stars_calc_masks(
        detection_image=detection_image, sex_path=sex_path, 
        input_sex_config=input_config, output_sex_parameters=output_parameters, 
        lupton_rgb=lupton_rgb, size=size, class_star=class_star, 
        prefix_filename=prefix_filename, save_fig=False, estimate_fwhm=estimate_fwhm,
        verbose=verbose, 
    )

    unmask_sexstars = True
    unmask_stars = []
    while unmask_sexstars:
        in_opt = input('(UN)mask SExtractor stars? [(Y)es|(r)edo|(n)o|(q)uit]:').lower()
        if in_opt == 'y':
            newindx = input('type (space separated) the detections numbers to be unmasked: ')
            unmask_stars += [int(i) for i in newindx.split()]
            print_level(f'Current stars numbers are: {unmask_stars}')
            unmask_sexstars = True
        elif in_opt == 'r':
            unmask_stars = []
        elif in_opt == 'n' or in_opt == '':
            unmask_stars = []
            unmask_sexstars = False
            # save figure
            fig_filename = f'{prefix_filename}_maskMosaic.png'
            print_level(f'Saving fig to {fig_filename}', 1, verbose)
            fig.savefig(fig_filename, format='png', dpi=180)
            plt.close(fig)
        elif in_opt == 'q':
            Warning('Exiting!')
            sys.exit(1)
        else:
            raise IOError('Option %s not recognized' % in_opt)
        if len(unmask_stars) or in_opt == 'r':
            resulting_mask, fig = sex_mask_stars_calc_masks(
                detection_image=detection_image, sex_path=sex_path, 
                input_sex_config=input_config, output_sex_parameters=output_parameters, 
                lupton_rgb=lupton_rgb, size=size, class_star=class_star, 
                prefix_filename=prefix_filename, save_fig=False, unmask_stars=unmask_stars, 
                estimate_fwhm=estimate_fwhm, verbose=verbose, fig=fig,
            )
    return resulting_mask
    
def sex_mask_stars():    
    from .headers import get_author, get_key
    from .constants import SPLUS_DEFAULT_SEXTRACTOR_CONFIG, SPLUS_DEFAULT_SEXTRACTOR_PARAMS
    
    from .utilities.splusdata import connect_splus_cloud, detection_image_hdul

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

    h = hdul[1].header

    mask_filename = detection_image.replace('detection', 'mask')

    input_config = SPLUS_DEFAULT_SEXTRACTOR_CONFIG
    input_config.update({
        'DETECT_THRESH': args.detect_thresh,
        'SATUR_LEVEL': args.satur_level,
        'GAIN': h.get(get_key('GAIN', get_author(h))),
        'SEEING_FWHM': h.get(get_key('PSFFWHM', get_author(h))),
        'BACK_SIZE': args.back_size,
        'CHECKIMAGE_NAME': detection_image.replace('detection', 'segmentation'),
    })
    output_parameters = SPLUS_DEFAULT_SEXTRACTOR_PARAMS
    
    lupton_rgb = _get_lupton_RGB(conn, args, save_img=False)

    resulting_mask = sex_mask_stars_loop_mask(
        sex_path=args.sextractor, 
        detection_image=detection_image, 
        lupton_rgb=lupton_rgb, 
        input_config=input_config, 
        output_parameters=output_parameters, 
        prefix_filename=prefix_filename, 
        size=args.size, 
        class_star=args.class_star, 
        estimate_fwhm=args.estimate_fwhm,
        verbose=0,
    )
    mhdul = hdul.copy()
    mhdul[1].data = resulting_mask
    mhdul[1].header['IMGTYPE'] = ('MASK', 'boolean mask')
    del mhdul[1].header['EXPTIME']
    del mhdul[1].header[get_key('GAIN', get_author(mhdul[1].header))]
    print_level(f'Saving mask to {mask_filename}')
    mhdul.writeto(mask_filename, overwrite=True)