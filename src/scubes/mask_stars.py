import sys
from os import remove
from os.path import join
from astropy.wcs import WCS
from astropy.io import fits
from .control import control
from matplotlib import pyplot as plt
from .headers import get_key, get_author
from .constants import SPLUS_DEFAULT_SEXTRACTOR_CONFIG, SPLUS_DEFAULT_SEXTRACTOR_PARAMS

from .utilities.io import print_level
from .utilities.plots import plot_mask
from .utilities.stats import robustStat
from .utilities.daofinder import DAOregions
from .utilities.sextractor import unmask_sewregions,SEWregions, run_sex

class _control(control):
    '''
    Subclass of `control` for managing specific settings.
    
    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :function:`argparse.ArgumentParser.parse_args()` 
        for control and configuration.
    
    output_dir : str, optional
        Output directory to save results. Default is '.'.
    '''    
    def __init__(self, args, output_dir='.'):
        super().__init__(args)
        self.output_dir = output_dir

class maskStars:
    '''
    Class for performing star masking on an image.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments parsed by :function:`argparse.ArgumentParser.parse_args()` 
        for control and configuration.

    detection_image : str
        Detection image where stars are detected.

    lupton_rgb : str
        Lupton RGB image for display.

    output_dir : str, optional
        Output directory to save results. Default is '.'.
    '''
    def __init__(self, args, detection_image, lupton_rgb, output_dir='.'):
        self.control = _control(args)
        self.detection_image = detection_image
        self.output_dir = output_dir
        self.lupton_rgb = lupton_rgb
        self.fig = None
        self.mask()

    def calc_masks(self, input_config, output_parameters, unmask_stars=None, run_DAOfinder=False, save_fig=False):
        '''
        Calculate stars mask using SExtractor. If `self.control.estimate_fwhm` 
        is True will run the mask two times in order to estimate the Full width 
        at half maximum (FWHM) of the detection image.

        Parameters
        ----------
        input_config : dict
            Settings for SExtractor.

        output_parameters : dict
            Output parameters for SExtractor.

        unmask_stars : list of int, optional
            List of star indices to unmask.

        run_DAOfinder : bool, optional
            Run DAOregions. Default is False.

        save_fig : bool, optional
            Save the generated figure. Default is False.

        Returns
        -------
        resulting_mask : ndarray
            Resulting mask.
        '''
        ctrl = self.control

        print_level('Calculating mask...')
        print_level('Running SExtractor to get photometry...')

        i = 0 if ctrl.estimate_fwhm else 1
        while i < 2:
            sewcat = run_sex(
                sex_path=ctrl.sextractor, 
                detection_fits=self.detection_image, 
                input_config=input_config, 
                output_params=output_parameters, 
                work_dir=self.output_dir, 
                output_file=self.detection_image.replace('detection', 'sexcat'), 
                verbose=ctrl.verbose,
            )
            if not i and ctrl.estimate_fwhm:
                stats = robustStat(sewcat['table']['FWHM_IMAGE']) 
                psffwhm = stats['median']*0.55
                fits.setval(self.detection_image, 'HIERARCH OAJ PRO FWHMMEAN', value=psffwhm, comment='', ext=1)
                files_to_remove = ['params.txt', 'conv.txt', 'config.txt', 'default.psf']
                for _f in files_to_remove:
                    remove(join(self.output_dir, _f))
                input_config['SEEING_FWHM'] = psffwhm
            i += 1
        
        h = fits.getheader(self.detection_image, ext=1)
        
        sewregions = SEWregions(sewcat=sewcat, class_star=ctrl.class_star, shape=(h.get('NAXIS2'), h.get('NAXIS1')), verbose=ctrl.verbose)
        
        data = fits.getdata(self.detection_image, ext=1)
        
        daoregions = None
        if run_DAOfinder:
            daoregions = DAOregions(data=data)
        
        masked_ddata, resulting_mask = unmask_sewregions(data=data, sewregions=sewregions, size=ctrl.size, unmask_stars=unmask_stars, verbose=ctrl.verbose)

        self.fig = plot_mask(
            detection_image=self.detection_image,
            lupton_rgb=self.lupton_rgb, 
            masked_ddata=masked_ddata, 
            resulting_mask=resulting_mask, 
            sewregions=sewregions, 
            daoregions=daoregions, 
            save_fig=save_fig,
            prefix_filename=join(self.output_dir, ctrl.prefix_filename),
            fig=self.fig,
        )
        return resulting_mask

    def loop_mask(self, input_config, output_parameters):
        '''
        Execute the interactive masking loop.

        Parameters
        ----------
        input_config : dict
            Settings for SExtractor.

        output_parameters : dict
            Output parameters for SExtractor.

        Returns
        -------
        resulting_mask : ndarray
            Resulting mask.
        '''        
        ctrl = self.control

        resulting_mask = self.calc_masks(input_config=input_config, output_parameters=output_parameters, unmask_stars=None, run_DAOfinder=False, save_fig=ctrl.no_interact)

        if not ctrl.no_interact:
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
                    fig_filename = join(self.output_dir, f'{ctrl.prefix_filename}_maskMosaic.png')
                    print_level(f'Saving fig to {fig_filename}', 1, ctrl.verbose)
                    self.fig.savefig(fig_filename, format='png', dpi=180)
                    plt.close(self.fig)
                elif in_opt == 'q':
                    Warning('Exiting!')
                    sys.exit(1)
                else:
                    raise IOError('Option %s not recognized' % in_opt)
                if len(unmask_stars) or in_opt == 'r':
                    resulting_mask = self.calc_masks(input_config, output_parameters, unmask_stars=unmask_stars, run_DAOfinder=False)
        return resulting_mask

    def mask(self):
        '''
        Perform star masking on the detection image.
        '''        
        ctrl = self.control
        mask_filename = self.detection_image.replace('detection', 'mask')
        
        hdul = fits.open(self.detection_image)
        h = hdul[1].header

        input_config = SPLUS_DEFAULT_SEXTRACTOR_CONFIG
        input_config.update({
            'DETECT_THRESH': ctrl.detect_thresh,
            'SATUR_LEVEL': ctrl.satur_level,
            'GAIN': h.get(get_key('GAIN', get_author(h))),
            'SEEING_FWHM': h.get(get_key('PSFFWHM', get_author(h))),
            'BACK_SIZE': ctrl.back_size,
            'CHECKIMAGE_NAME': self.detection_image.replace('detection', 'segmentation'),
        })
        output_parameters = SPLUS_DEFAULT_SEXTRACTOR_PARAMS

        resulting_mask = self.loop_mask(input_config=input_config, output_parameters=output_parameters)

        mhdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(resulting_mask.astype('int'))])

        # UPDATE HEADER
        w = WCS(h)
        mhdul[1].header.update(w.to_header())
        mhdul[1].header['IMGTYPE'] = ('MASK', 'boolean mask')
        mhdul[1].header['EXTNAME'] = ('MASK', 'Boolean mask of the galaxy')
        print_level(f'Saving mask to {mask_filename}')
        mhdul.writeto(mask_filename, overwrite=True)

        self.filename = mask_filename
        self.hdul = mhdul