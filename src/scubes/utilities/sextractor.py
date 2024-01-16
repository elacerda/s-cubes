import numpy as np
from importlib import resources
from regions import PixCoord, CirclePixelRegion

from .data import sex

import os

__data_files__ = sex.__path__[0]

SEX_TOPHAT_FILTER = os.path.join(__data_files__, 'tophat_3.0_3x3.conv') 
SEX_DEFAULT_FILTER = os.path.join(__data_files__, 'default.conv') 
SEX_DEFAULT_STARNNW = os.path.join(__data_files__, 'default.nnw') 

from sewpy import SEW
from os.path import join
from astropy.io import fits

from .io import print_level

def run_sex(sex_path, detection_fits, input_config, output_params, work_dir=None, output_file=None, overwrite=True, verbose=0):    
    '''
    Run SExtractor on the given FITS file.

    Parameters
    ----------
    sex_path : str
        Path to the SExtractor executable.
    
    detection_fits : str
        Path to the input FITS file for detection.
    
    input_config : str
        Path to the SExtractor configuration file.
    
    output_params : list
        List of output parameters for SExtractor.
    
    work_dir : str, optional
        Working directory for SExtractor. Defaults to None (current directory).
    
    output_file : str, optional
        Path to the output catalog FITS file. Defaults to None.
    
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Defaults to True.
    
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    sewpy.output.SExtractor
        SExtractor catalog result.
    '''    
    print_level('Running SExtractor for config:', 2, verbose)
    print_level(input_config, 2, verbose)

    sew = SEW(workdir='.' if work_dir is None else work_dir, config=input_config, sexpath=sex_path, params=output_params)
    sewcat = sew(detection_fits)
    
    if output_file is not None:
        sewcat['table'].write(output_file, format='fits', overwrite=overwrite)

    return sewcat

def SEWregions(sewcat, shape, class_star, verbose=0):
    '''
    Generate circular regions based on SExtractor output.

    Parameters
    ----------
    sewcat : sewpy.output.SExtractor
        SExtractor catalog result.
    
    shape : tuple
        Shape of the image (height, width).
    
    class_star : float
        CLASS_STAR threshold for star/galaxy separation.
    
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    list of regions.CirclePixelRegion
        List of circular regions.
    '''    
    print_level(f'Using CLASS_STAR > {class_star:.2f} star/galaxy separator...', 1, verbose)
    sewpos = np.transpose((sewcat['table']['X_IMAGE'], sewcat['table']['Y_IMAGE']))
    radius = 3.0 * (sewcat['table']['FWHM_IMAGE'] / 0.55)
    sidelim = 80
    mask = sewcat['table']['CLASS_STAR'] > class_star
    mask &= sewcat['table']['X_IMAGE'] > sidelim
    mask &= sewcat['table']['X_IMAGE'] < (shape[0] - sidelim)
    mask &= sewcat['table']['Y_IMAGE'] > sidelim
    mask &= sewcat['table']['Y_IMAGE'] < (shape[0] - sidelim)
    mask &= sewcat['table']['FWHM_IMAGE'] > 0
    return [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(sewpos[mask], radius[mask])]

def unmask_sewregions(data, sewregions, size, unmask_stars=None, verbose=0):
    '''
    Unmask circular regions in the given data.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data.
    
    sewregions : list of regions.CirclePixelRegion
        List of circular regions.
    
    size : int
        Size of the image.
    
    unmask_stars : list, optional
        List of indices to exclude from unmasking. Defaults to None.
    
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        Tuple containing masked data (numpy.ndarray) and resulting mask (numpy.ndarray).
    '''    
    unmask_stars = [] if unmask_stars is None else unmask_stars
    stars_mask = np.ones(data.shape)
    for n, sregion in enumerate(sewregions):
        if n not in unmask_stars:
            mask = sregion.to_mask()
            if (min(mask.bbox.extent) < 0) or (max(mask.bbox.extent) > size):
                print_level(f'Region is out of range for extent {mask.bbox.extent}')
            else:
                _slices = (slice(mask.bbox.iymin, mask.bbox.iymax), slice(mask.bbox.ixmin, mask.bbox.ixmax))
                print_level(f'{mask.bbox.extent} min: {min(mask.bbox.extent)} {_slices}', 2, verbose)
                stars_mask[_slices] *= 1 - mask.data
    # TODO: CHANGE TO BOOL ?
    stars_mask = np.where(stars_mask == 1, 0, 2)
    #resulting_mask = detection_mask + stars_mask
    resulting_mask = stars_mask.astype('bool')
    #masked_data = np.where(resulting_mask > 0, 0, data)
    masked_data = np.where(resulting_mask, 0, data)
    return masked_data, resulting_mask
