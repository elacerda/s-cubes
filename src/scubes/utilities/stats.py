from os.path import join
from astropy.io import fits
from .sextractor import run_sex

def estimate_fwhm(sex_path, detection_fits, data_path, work_dir=None, output_file=None, verbose=0):

    params = [
        'NUMBER', 'X_IMAGE', 'Y_IMAGE',
        'THETA_WORLD', 'ERRTHETA_WORLD', 'ERRA_IMAGE', 'ERRB_IMAGE',
        'ERRTHETA_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO',
        'ALPHA_J2000', 'DELTA_J2000', 'X_WORLD', 'Y_WORLD',
        'MAG_AUTO', 'MAG_BEST', 'MAGERR_AUTO',
        'MAGERR_BEST', 'FLUX_MAX', 'FWHM_IMAGE', 'FLAGS',
        'ELLIPTICITY','MU_THRESHOLD', 'THRESHOLD', 'BACKGROUND', 'THETA_IMAGE',
        'A_IMAGE', 'B_IMAGE','FLUX_RADIUS','ISOAREA_IMAGE'
    ]
    h = fits.getheader(detection_fits, ext=1)

    config = {
        'BACK_FILTERSIZE': 3,
        'BACK_SIZE': 256,
        'BACK_TYPE': 'AUTO',
        'BACK_VALUE': '0.0,0.0',
        'THRESH_TYPE': 'RELATIVE',
        'ANALYSIS_THRESH': 1.5,
        'DETECT_THRESH': 1.5,
        'DETECT_MINAREA': 5,
        'FILTER_THRESH': '',
        'FILTER': 'Y',
        'FILTER_NAME': join(data_path, 'default.conv'),
        'CLEAN': 'Y',
        'CLEAN_PARAM': 1.0,
        'DEBLEND_NTHRESH': 32,
        'DEBLEND_MINCONT': 0.005,
        'MASK_TYPE': 'CORRECT',
        'WEIGHT_TYPE': 'BACKGROUND',
        'WEIGHT_IMAGE': join(work_dir, 'estfwhm_weight.fits'),
        'WEIGHT_THRESH': '',
        'WEIGHT_GAIN': 'Y',
        'GAIN': h.get('GAIN'),
        'GAIN_KEY': 'GAIN',
        'FLAG_IMAGE': 'flag.fits',
        'FLAG_TYPE': 'OR',
        'BACKPHOTO_TYPE': 'LOCAL',
        'BACKPHOTO_THICK': 24,
        'BACK_FILTTHRESH': 0.0,
        'PHOT_AUTOPARAMS': (2.5, 3.5),
        'PHOT_AUTOAPERS': (0.0, 0.0),
        'PHOT_PETROPARAMS': (2.0, 3.5),
        'PHOT_APERTURES': (2, 28, 160),
        'PHOT_FLUXFRAC': 0.5,
        'SATUR_LEVEL': h.get('SATURATE'),
        'SATUR_KEY': 'SATURATE',
        'STARNNW_NAME': join(data_path, 'default.nnw'),
        'SEEING_FWHM': 1.1,
        'CATALOG_NAME': join(work_dir, 'estfwhm.cat'),
        'PARAMETERS_NAME': join(work_dir, 'estfwhm_params.sex'),
        'CHECKIMAGE_TYPE': 'SEGMENTATION, APERTURES, OBJECTS',
        'CHECKIMAGE_NAME': f"{join(work_dir, 'estfwhm_SEGM.fits')}, {join(work_dir, 'estfwhm_APER.fits')}, {join(work_dir, 'estfwhm_OBJ.fits')}",
        'INTERP_TYPE': 'NONE',
        'INTERP_MAXYLAG': 4,
        'INTERP_MAXXLAG': 4,
        'DETECT_TYPE': 'CCD',
        'MEMORY_BUFSIZE': 11000,
        'MEMORY_PIXSTACK': 3000000,
        'MEMORY_OBJSTACK': 10000,
        'PIXEL_SCALE': 0.55,
        'MAG_GAMMA': 4.0,
        'MAG_ZEROPOINT': 25.0,
        'CATALOG_TYPE': 'FITS_LDAC',
        'VERBOSE_TYPE': 'NORMAL',
        'WRITE_XML': 'Y',
        'XML_NAME': join(work_dir, 'sexout.xml'),
        'NTHREADS': 2,
    }

    sewcat = run_sex(
        sex_path=sex_path,
        detection_fits=detection_fits,
        input_config=config,
        output_params=params,
        work_dir=work_dir,
        output_file=output_file,
        verbose=verbose,
    )




   