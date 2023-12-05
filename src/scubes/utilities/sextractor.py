from sewpy import SEW
from .io import print_level

def run_sex(sex_path, detection_fits, input_config, output_params, work_dir=None, output_file=None, overwrite=True, verbose=0):
    print_level('Running SExtractor for config:', 2, verbose)
    print_level(input_config, 2, verbose)

    sew = SEW(workdir='.' if work_dir is None else work_dir, config=input_config, sexpath=sex_path, params=output_params)
    sewcat = sew(detection_fits)
    
    if output_file is not None:
        sewcat['table'].write(output_file, format='fits', overwrite=overwrite)

    return sewcat