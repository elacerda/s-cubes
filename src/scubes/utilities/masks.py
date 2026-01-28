import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from regions import PixCoord, CirclePixelRegion 
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder, StarFinder
from photutils.segmentation import make_2dgaussian_kernel, \
    detect_sources, deblend_sources

from .io import print_level
from .psf import calc_PSF_scube
from .plots import plot_masks_psf, plot_extra_sources, plot_scube_RGB_mask_sky, \
        plot_violin_reescaled_error_spectrum, plot_masks_final_plot

def _star_fwhm_calc_apertures(data, centers, star_peak, individual=False, star_threshold=50, worst_psf=3, star_size_calc='sky', med_sqrt=False, save_plot=None):
    mean, _, std = sigma_clipped_stats(data)
    if individual:
        psf__bsxy = calc_PSF_scube(data, centers_xy=centers, med_sqrt=med_sqrt, save_plot=save_plot)
        psf = np.ma.median(np.ma.sqrt(psf__bsxy[:, :, 1]**2 + psf__bsxy[:, :, 0]**2), axis=0)
        norm = 2*np.sqrt(2*np.log(2))
        sig = np.where(psf.mask, worst_psf/norm, psf/norm)
        if star_size_calc == 'sky':
            ratios = np.sqrt(-2*sig**2*np.log(mean/(star_peak)))
        else:
            ratios = star_size_calc*np.where(psf.mask, worst_psf, psf)
    else:
        sig = worst_psf/norm
        ratios = np.sqrt(-2*sig**2*np.log(std/(star_threshold*star_peak)))
    return [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(centers, ratios)]

def clean_mask_polygons(mask, largest=True, nmin=2, masked_value=False):
    """
    Removes polygons (islands) from a mask (image)

    Parameters
    ----------
    mask : array
        2-D image where the ``True`` pixels mark the data.

    largest : bool
        Get the largest polygon in the image and remove everything else

    nmin : integer
        If ``largest`` is False, ``nmin`` is the minimum number of pixels (size) of the
        polygons to keep. Polygons with less pixels than ``nmin`` will be removed

    masked_value : float, integer, np.ma.masked
        Value to fill the removed polygons of ``mask``

    Returns
    -------
    cleaned_mask : array
        2-D image of same shape and type as ``mask``,
        where the values of remain polygons of ``mask``
        are marked as ``True`` values.
    """
    import scipy.ndimage
    mask = mask.copy()
    labels, num = scipy.ndimage.label(mask)
    objects = scipy.ndimage.find_objects(labels)
    size = np.zeros(num)
    for i, islice in enumerate(objects):
        size[i] = len(mask[islice].flatten())
    nmax = size.max() if largest else nmin
    for isize, islice in zip(size, objects):
        if isize < nmax:
            mask[islice] = masked_value
    return mask

class masks_builder:
    def __init__(self, scube, args):
        self.scube = scube        
        self.args = args   
        self._init_vars()
    
    def _init_vars(self):
        self.mask_has_inf_err__yx = np.isinf(self.scube.eflux__lyx).sum(axis=0) > 0
        self.mask_stars__yx = None
        self.mask_stars_pos__xy = None
        self.iso_mask__yx = None
        self.tot_mask_yx = None
        self.final_mask__yx = None
        self.sky = None
        self.mask_sky__yx = None

    def rescaled_error_spectrum(self):
        fm = self.final_mask__yx
        scube = self.scube
        if (fm is None) or (scube.eflux__lyx is None):
            return None, None
        # Compute "Re-scaled" error spectrum & add to hdu "info" for later output
        errR__byx = scube.eflux__lyx/scube.eflux__lyx[7]
        errR__byx = np.ma.masked_where(~np.isfinite(errR__byx), errR__byx)
        errRel_meanSpec__b = np.ma.mean(errR__byx[:, (fm == 0)], axis=1)
        self.errR__byx = errR__byx
        self.errRel_meanSpec__b = errRel_meanSpec__b
        return self.errR__byx, self.errRel_meanSpec__b

    def tot_mask(self):
        m = self.mask_stars__yx
        i = self.iso_mask__yx
        if (m is None) or (i is None):
            return None
        self.tot_mask__yx = np.where(m == 1, 1, 2*i.astype(int))
        return self.tot_mask__yx
    
    def final_mask(self):
        # Def final_mask__yx = 0 (gal), 1 (stars), 2 (sky), 99 (shit)
        # OBS: This was just: final_mask__yx = np.where(wei__yx == 0, tot_mask__yx, 99)
        #      but now we've added the infinite errors to the final mask!
        scube = self.scube
        w = scube.weimask__yx
        i = self.mask_has_inf_err__yx
        m = self.tot_mask__yx
        if (i is None) or (m is None):
            return None
        self.final_mask__yx = np.where((w == 0) & (~i), m, 99)
        return self.final_mask__yx
    
    def isophot_mask(self, band=7, isophotal_limit=24, isophotal_medsize=30, stars_mask=False, clean_polygons=False):       
        refmag__yx = self.scube.mag_arcsec2__lyx[
            self.scube.get_filter_i(filt=band) if not isinstance(band, int) else band
        ]
        _tmp = ~((refmag__yx <= isophotal_limit) & (stars_mask == 0))
        iso_mask = median_filter(_tmp, size=isophotal_medsize) | (stars_mask == 1)
        if clean_polygons:
            iso_mask = ~clean_mask_polygons(~iso_mask)
        self.iso_mask__yx = iso_mask
        return iso_mask
    
    def mask_sky(self, ref_mag_filt=7, isophotal_limit=25, isophotal_medsize=10, stars_mask=None, n_sigma=3, n_iter=5, clip_neg=False):
        '''
        A Wrapper to :meth:`read_scube.get_iso_sky`. 
        '''
        self.sky, self.mask_sky__yx = self.scube.get_iso_sky(
            ref_mag_filt=ref_mag_filt,
            isophotal_limit=isophotal_limit,
            isophotal_medsize=isophotal_medsize,
            stars_mask=stars_mask,
            n_sigma=n_sigma,
            n_iter=n_iter,
            clip_neg=clip_neg            
        )

    def build_star_mask(self, 
                        bands=7, star_threshold=50, worst_psf=3, xsource_std_f=3, 
                        mask_Ha=True, extra_mask__yx=False, 
                        detection='StarFinder', 
                        star_fwhm_individual_calc=True, star_size_calc='sky', 
                        save_fig=None, 
                        Q=3, stretch=130, im_max=180, minimum=15, plim=0.3, 
                        check_sources=True, no_interact=False):
        """
        Build masks of stars in front of the galaxy
        * if noise image, change the pLim
        detection= 'StarFinder' ou 'DAOStarFinder'

        Using DAOFIND:
            Detect stars in an image using the DAOFIND (`Stetson 1987
            <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
            algorithm.

        Using StarFinder:
            Detect stars in an image using a user-defined kernel. From photutils.

        Ex.:
        bands = (5,7,9) -> what bands are used in the image to find stars. If it is more than one, is made the sum.
        mask_Ha = True -> Mask of Ha region using the RGB image. [Obs.: for now (3FM didn't work as well as)]
        (First version: Q= 3.5, stretch=100, im_max=120, minimum=15, plim=0.3)
        worst_psf: is the FWHM used to calculate the size of the stars (in pixels)
        # The full-width half-maximum (FWHM) of the major axis of the Gaussian kernel in units of pixels.
        :threshold: threshold is the value that will be multiplied by the std(sky) to be used as threshold to find stars
        :xsource_std_f: factor multiplied by the standard deviation to be used as threshold to extra source
        :check_sources: whether True will plot the detected sources and ask for changes; False will not, just return
        :star_fwhm_individual_calc: calculate the FWHM of each star
        :star_size_calc: choose how is calculated the "size" of the star
            -> 'sky' : the size is defined when the flux == sky
            -> Number: factor that multiply the FWHM to get the size of each star
        :save_fig: path+name to save the plot with the detected stars, if None is not saved
        :return: star mask__yx array and the positions__xy of the stars respect

        Júlia Granada 28/11/23 - 01/24
        Update Júlia Granada 27/02/24 - default parameters changed from (Q= 3.5, stretch=100, im_max=120, minimum=15, plim=0.3)
                    to (Q= 3, stretch=130, im_max=180, minimum=15, plim=1.5) and RGB changed from j0660,r,g to j0660,i,g
                    RGB = 8, 9, 7 also gets a good map
        Update Júlia Granada 07/06/24 - add parameter star_fwhm_individual_calc
        Update Eduardo@RV 17/02/2025 - convert to :class:`scubes.readscube.read_scube` usage.
        """     
        import matplotlib.pyplot as plt
        plt.ion()

        if detection not in ['DAOStarFinde', 'StarFinder']:
            print_level('detection should be DAOStarFinder or StarFinder!!')
            sys.exit()

        # sum all data to a single 2D image in order to detect the sources
        scube = self.scube

        if type(bands) is int:
            data__yx = np.copy(scube.flux__lyx[bands])
        else:
            data__yx = np.copy(scube.flux__lyx[bands, :, :].sum(axis=0))
        
        # use sigma-clipped statistics to (roughly) estimate the background noise levels
        mean, median, std = sigma_clipped_stats(data__yx)

        # define the center and exclude the center
        ny, nx = data__yx.shape
        center = ((nx - 1)/2, (ny - 1)/2)
        ap_center = CircularAperture(center, r=max(min(nx/100, 10), 2.5)) # change with size of the galaxy
        mask_center = ap_center.to_mask().to_image(shape=data__yx.shape)
        mask_center = np.ma.masked_where(mask_center == 1, mask_center)

        # Ha mask with R, G and B limits
        pminmax = plim, 100 - plim
        _m = (minimum, minimum, minimum)
        RGB__yxc = scube.lRGB_image(rgb=(8, 9, 5), Q=Q, stretch=stretch, im_max=im_max, minimum=_m, pminmax=pminmax)
        if mask_Ha:
            mask_Ha = (RGB__yxc[:, :, 0] > 80)
            mask_Ha *= (RGB__yxc[:, :, 0] > (RGB__yxc[:, :, 1] + 30))
            mask_Ha *= (RGB__yxc[:, :, 0] > (RGB__yxc[:, :, 2] + 30))
            mask_Ha *= (RGB__yxc[:, :, 1] < 210)
            mask_Ha *= (RGB__yxc[:, :, 2] < 210)
            mask_Ha *= (data__yx > 3*std)
            mask_Ha = median_filter(mask_Ha, 3)

        # joint masks
        mask_tot__yx = mask_center.mask + mask_Ha + extra_mask__yx + median_filter(mask_Ha, 5)
        
        # detect stars using a catalog
        sources = None
        if detection == 'DAOStarFinder':
            # threshold : The absolute image value above which to select sources.
            daofind = DAOStarFinder(fwhm=3.0, threshold=star_threshold*std)
            # subtract the background data__yx -= mean
            sources = daofind(data__yx - mean, mask=mask_tot__yx != 0)  # get the sources
            k_max = 'peak'
        elif detection == 'StarFinder':
            kernel = make_2dgaussian_kernel(worst_psf, size=5)
            finder = StarFinder(threshold=star_threshold*std, kernel=kernel.array)
            sources = finder.find_stars(data__yx - mean, mask=mask_tot__yx != 0)  # get the sources
            k_max = 'max_value'
        if sources is not None:
            f_max = sources[k_max]
        else:
            # GAME OVER!
            mask_stars__yx = np.zeros_like(data__yx)  # ADD EXTRA SOURCE HERE ??
            positions__xy = []
            self.mask_stars__yx = mask_stars__yx
            self.mask_stars_pos__xy = positions__xy
            return mask_stars__yx, positions__xy
    
        # get the positions of the sources x,y!!!!
        positions__xy = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = _star_fwhm_calc_apertures(
            data=scube.flux__lyx, 
            centers=positions__xy,
            star_peak=f_max, 
            individual=star_fwhm_individual_calc, 
            star_threshold=star_threshold, 
            worst_psf=worst_psf, 
            star_size_calc=star_size_calc, 
            med_sqrt=False, 
            save_plot=save_fig
        )
        
        # from positions to mask
        mask_stars__yx = np.zeros(shape=data__yx.shape)
        for ap in apertures:
            m_ap = ap.to_mask().to_image(shape=data__yx.shape)
            mask_stars__yx += m_ap
        mask_stars__yx = np.where(mask_stars__yx > 1, 1, mask_stars__yx)

        if check_sources:
            fig_masks, ax_masks = plot_masks_psf(RGB__yxc, apertures, mask_Ha, mask_stars__yx)
            no_mask = ''
            if not no_interact:
                no_mask = input('UNmask any source? \nNo: just enter! \nYes: write the numbers (ex.: 2, 54): ')
                extra_source = input('Extra source? ([no] or yes) ').lower().strip()
                if (extra_source == 'y') or (extra_source == 'yes'):
                    # detect the sources and do the segmentation
                    segm = detect_sources(data__yx - mean, std, npixels=30)
                    extra_source = deblend_sources(data__yx - mean, segm, npixels=10, mode='linear')
                    fig_xsource, ax_xsource = plot_extra_sources(extra_source, filename=save_fig)
                    str_sources = input('Which sources? (ex.: 31, 42)\n')
                    if str_sources:
                    #xsources = str_sources.split(',')
                    #n_xsources = len(xsources)
                    #if n_xsources:
                        for n in str_sources.split(','):
                            mask_stars__yx = np.where(extra_source.data == int(n), 1, mask_stars__yx)
                        extra_source2 = input('Separate extra source? ([no] or yes) ').lower()
                        if (extra_source2 == 'y') or (extra_source2 == 'yes'):
                            # z1, z2 = int(min(data__yx.shape) * 0.2), int(min(data__yx.shape) * 0.8)
                            # mean, median, std = sigma_clipped_stats(data__yx[z1:z2, z1:z2])
                            # segm = detect_sources(data__yx[z1:z2, z1:z2] - mean, std, npixels=30)
                            segm = detect_sources(data__yx - mean, xsource_std_f*std, npixels=30)
                            extra_source2 = deblend_sources(data__yx - mean, segm, npixels=10, mode='exponential')
                            fig_xsource2, ax_xsource2 = plot_extra_sources(extra_source2, filename=save_fig)
                            if extra_source2 == extra_source:
                                print_level("Not possible separate any source.")
                            else:
                                str_sources = input('Which sources? (ex.: 31, 42)\n')
                                if str_sources:
                                #xsources = str_sources.split()
                                #n_xsources = len(xsources)
                                #if n_xsources:
                                    for n in str_sources.split(','):
                                        mask_stars__yx = np.where(extra_source2.data == int(n), 1, mask_stars__yx)
            if len(no_mask):
                for i in no_mask.split(','):
                    mask_stars__yx -= apertures[int(i)].to_mask().to_image(shape=data__yx.shape)
                fig_masks, ax_masks = plot_masks_psf(RGB__yxc, apertures, mask_Ha, mask_stars__yx, fig=fig_masks)
            ax_masks[-1].imshow(np.ma.masked_where(mask_stars__yx == 0, mask_stars__yx), cmap='gray_r', origin='lower',interpolation='nearest')
            if save_fig is not None:
                fig_masks.savefig(save_fig)   
        self.mask_stars__yx = mask_stars__yx 
        self.mask_stars_pos__xy = positions__xy    
        return mask_stars__yx, positions__xy

    def cut_cube_wcs(self, mask_gal__yx, extra_pix=5):
        """
        !OJO! Better doc!
        Cut the cube
        mask_gal__yx: the mask galaxy where gal==0
        """
        scube = self.scube
        # Def region to cut
        ny, nx = mask_gal__yx.shape
        cx, cy = (nx - 1)/2, (ny - 1)/2  # Center for cropping
        size_gal = np.abs(np.argwhere(mask_gal__yx == 0) - np.array([cy, cx])).max()
        cutx = (cx - (size_gal + extra_pix)).astype(int)  # Pixels to cut from each side -> center - size radial of the mask
        cuty = (cy - (size_gal + extra_pix)).astype(int)

        # Load the data and WCS from the header
        flux_cube__byx = scube.flux__lyx
        error_cube__byx = scube.eflux__lyx
        wcs_3d = WCS(scube.data_header)

        # Perform Cutout for the mask
        mask_cutout__yx = Cutout2D(mask_gal__yx, (cx, cy), (ny - 2 * cuty, nx - 2 * cutx), wcs=scube.wcs)

        flux_cut__byx = flux_cube__byx[:, cuty:-cuty, cutx:-cutx]
        err_cut__byx = error_cube__byx[:, cuty:-cuty, cutx:-cutx]

        # Update the 3D WCS using mask_cutout__yx.wcs
        wcs_3d.wcs.crpix[0] = mask_cutout__yx.wcs.wcs.crpix[0]  # Updated CRPIX for x-axis
        wcs_3d.wcs.crpix[1] = mask_cutout__yx.wcs.wcs.crpix[1]  # Updated CRPIX for y-axis
        wcs_3d.array_shape = (flux_cut__byx.shape[0], mask_cutout__yx.data.shape[0], mask_cutout__yx.data.shape[1])

        # Create headers with updated WCS for flux and error
        flux_header = wcs_3d.to_header()
        # flux_header['NAXIS1'] = mask_cutout__yx.data.shape[1]  # X-dimension of the cutout
        # flux_header['NAXIS2'] = mask_cutout__yx.data.shape[0]  # Y-dimension of the cutout
        # flux_header['NAXIS3'] = flux_cut__byx.shape[0]         # Number of slices (unchanged)
        flux_header['BUNIT'] = ('erg / (Angstrom s cm2)', 'Physical units of the array values')
        err_header = flux_header.copy()
        err_header['EXTNAME'] = 'error'
        flux_header['EXTNAME'] = 'flux'

        # Use the updated WCS header from Cutout2D for the mask
        mask_header = mask_cutout__yx.wcs.to_header()
        mask_header['EXTNAME'] = 'mask'

        # Create HDUs
        hdu_flux = fits.ImageHDU(flux_cut__byx, header=flux_header)
        hdu_error = fits.ImageHDU(err_cut__byx, header=err_header)
        hdu_mask = fits.ImageHDU(mask_cutout__yx.data, header=mask_header)

        return hdu_mask, hdu_flux, hdu_error
    
    def build_final_hdul(self):
        from .utils import SCUBE_MASK_ARGS

        scube = self.scube

        orig_table = scube._hdulist[4].data
        orig_cols = orig_table.columns
        new_cols = fits.ColDefs([
            fits.Column(name='mean_sky', array=self.sky['mean__l'], format='D', unit='flux'),
            fits.Column(name='errRel_meanSpec__b', array=self.errRel_meanSpec__b, format='D', unit='norm by r band'),
        ])
        hdu_info = fits.BinTableHDU.from_columns(orig_cols + new_cols)
        hdu_info.name = 'FilterInfo'

        _ = self.cut_cube_wcs(self.final_mask__yx, extra_pix=self.args.extra_pix)
        hdu_final_mask__yx, hdu_flux__byx, hdu_err__byx = _

        hdu_final_mask__yx.header['label'] = 'stars=1, sky=2, galaxy=0, weimask=99'

        #Adding pix scale to mask header
        hdu_final_mask__yx.header.append(('PIXSCALE', scube.pixscale))
        
        # Adding input parameters to mask header
        for k in SCUBE_MASK_ARGS:
            v = getattr(self.args, k)
            if isinstance(v, list):
                v = str(v)
            hdu_final_mask__yx.header.append((f'HIERARCH {k}', v, 'parameter of scube_mask script'))

        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(header=scube.primary_header), 
                hdu_flux__byx, 
                hdu_err__byx, 
                hdu_final_mask__yx, 
                hdu_info
            ]
        )

        self.hdul = hdul.copy()
        hdul.writeto(f'{scube.galaxy}.fits', overwrite=True)
        print_level('Saving cut cube')
        hdul.close()

    def mask_procedure(self):
        from matplotlib import pyplot as plt
        
        scube = self.scube
        args = self.args

        self.build_star_mask(
            star_threshold=args.mask_stars_threshold, 
            bands=args.mask_stars_bands, 
            worst_psf=(scube.psf_fwhm/scube.pixscale).max(), 
            xsource_std_f=args.xsource_std_f, 
            star_size_calc=args.star_size_calc,
            save_fig=f'{scube.galaxy}_build_stars_mask.png',
            star_fwhm_individual_calc=args.star_fwhm_individual_calc,
            no_interact=args.no_interact,
        )
        self.isophot_mask(
            band=7, 
            isophotal_limit=args.mask_isophotal_limit, 
            isophotal_medsize=args.mask_isophotal_medsize, 
            stars_mask=self.mask_stars__yx, 
            clean_polygons=True
        )
        tmask__yx = self.tot_mask()
        if tmask__yx is not None:
            fmask__yx = self.final_mask()
        if fmask__yx is not None:
            self.mask_sky(
                ref_mag_filt=7,
                isophotal_limit=args.sky_isophotal_limit, 
                isophotal_medsize=args.sky_isophotal_medsize, 
                stars_mask=(self.final_mask__yx != 2), 
                n_sigma=args.sky_n_sigma, 
                n_iter=args.sky_n_iter, 
                clip_neg=args.sky_clip_neg
            )
        if self.sky is not None:
            plot_scube_RGB_mask_sky(scube, self)
            self.rescaled_error_spectrum()               
            plot_violin_reescaled_error_spectrum(scube, self)
            self.build_final_hdul()
            plot_masks_final_plot(scube, self)

        if not args.no_interact:
            input('...press any key to close the plots...')
        plt.close('all')
