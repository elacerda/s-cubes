from astropy.io import ascii
from .utilities.sextractor import SEX_TOPHAT_FILTER, SEX_DEFAULT_STARNNW

from . import __filters_table__

# This dictionary links the original filter names to the ones recorded on the 
# original S-PLUS image fits
FILTER_NAMES_FITS = {
    'J0378': 'F378', 'J0395': 'F395', 'J0410': 'F410', 'J0430': 'F430',
    'J0515': 'F515', 'J0660': 'F660', 'J0861': 'F861', 
    'uJAVA': 'U', 'gSDSS': 'G', 'rSDSS': 'R', 'iSDSS': 'I', 'zSDSS': 'Z',
}

FILTER_NAMES = {v: k for k, v in FILTER_NAMES_FITS.items()}

FILTER_NAMES_ZP_TABLE = {
    'F378': 'J0378', 'F395': 'J0395', 'F410': 'J0410', 'F430': 'J0430',
    'F515': 'J0515', 'F660': 'J0660', 'F861': 'J0861', 
    'U': 'u', 'G': 'g', 'R': 'r', 'I': 'i', 'Z': 'z',
}

BANDS = [FILTER_NAMES_FITS[x] for x in __filters_table__['filter']]

# EFF is the same as the pivot
CENTRAL_WAVE = {FILTER_NAMES_FITS[row['filter']]: row['pivot_wave'] for row in __filters_table__}

METADATA_NAMES = {
    'filter': 'FILTER',
    'central_wave': 'CENTWAVE',
    'delta_wave': 'DELTWAVE',
    'trapz_wave': 'TRAPWAVE',
    'trapz_width': 'TRAPWIDTH',
    'mean_wave': 'MEANWAVE',
    'mean_width': 'MEANWIDTH',
    'mean_1_wave': 'MEAN1WAVE',
    'mean_1_width': 'MEAN1WIDTH',
    'pivot_wave': 'PIVOTWAVE',
    'alambda_av': 'ALAMBDAAV',
}

EXPTIMES = {
    'F378': 660, 'F395': 354, 'F410': 177, 'F430': 171,  
    'F515': 183, 'F660': 870, 'F861': 240,
    'U': 681, 'G': 99, 'R': 120, 'I': 138, 'Z': 168,
}

SPLUS_DEFAULT_SEXTRACTOR_CONFIG = {
    'FILTER_NAME': SEX_TOPHAT_FILTER,
    'STARNNW_NAME': SEX_DEFAULT_STARNNW,
    'DETECT_TYPE': 'CCD',
    'DETECT_MINAREA': 4,
    'ANALYSIS_THRESH': 3.0,
    'FILTER': 'Y',
    'DEBLEND_NTHRESH': 64,
    'DEBLEND_MINCONT': 0.0002,
    'CLEAN': 'Y',
    'CLEAN_PARAM': 1.0,
    'MASK_TYPE': 'CORRECT',
    'PHOT_APERTURES': 5.45454545,
    'PHOT_AUTOPARAMS': '3.0,1.82',
    'PHOT_PETROPARAMS': '2.0,2.73',
    'PHOT_FLUXFRAC': '0.2,0.5,0.7,0.9',
    'MAG_ZEROPOINT': 20,
    'MAG_GAMMA': 4.0,
    'PIXEL_SCALE': 0.55,
    'BACK_FILTERSIZE': 7,
    'BACKPHOTO_TYPE': 'LOCAL',
    'BACKPHOTO_THICK': 48,
    'CHECKIMAGE_TYPE': 'SEGMENTATION',
}

SPLUS_DEFAULT_SEXTRACTOR_PARAMS = [
    'NUMBER', 'X_IMAGE', 'Y_IMAGE', 'KRON_RADIUS', 'ELLIPTICITY', 'THETA_IMAGE', 
    'A_IMAGE', 'B_IMAGE', 'MAG_AUTO', 'FWHM_IMAGE', 'CLASS_STAR'
]

#iDR4_FORNAX_RUN_7_106_Fornax_SPLUS-s28s33.00025, HYDRA_FULL+SPLUS-n17s10.00020, SPLUS-n16s09.00003
