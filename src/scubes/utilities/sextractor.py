from importlib import resources
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