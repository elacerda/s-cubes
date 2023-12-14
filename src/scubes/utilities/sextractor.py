import numpy as np
from importlib import resources
from regions import PixCoord, CirclePixelRegion

from .data import sex

__data_files__ = resources.files(sex)

SEX_TOPHAT_FILTER = str(__data_files__ / 'tophat_3.0_3x3.conv') 
SEX_DEFAULT_FILTER = str(__data_files__ / 'default.conv') 
SEX_DEFAULT_STARNNW = str(__data_files__ / 'default.nnw') 

from sewpy import SEW
from os.path import join
from astropy.io import fits

from .io import print_level

def run_sex(sex_path, detection_fits, input_config, output_params, work_dir=None, output_file=None, overwrite=True, verbose=0):
    print_level('Running SExtractor for config:', 2, verbose)
    print_level(input_config, 2, verbose)

    sew = SEW(workdir='.' if work_dir is None else work_dir, config=input_config, sexpath=sex_path, params=output_params)
    sewcat = sew(detection_fits)
    
    if output_file is not None:
        sewcat['table'].write(output_file, format='fits', overwrite=overwrite)

    return sewcat

def SEWregions(sewcat, shape, class_star, verbose=0):
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
    stars_mask = np.where(stars_mask == 1, 0, 2)
    #resulting_mask = detection_mask + stars_mask
    resulting_mask = stars_mask
    masked_data = np.where(resulting_mask > 0, 0, data)
    return masked_data, resulting_mask
