import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec    

from .io import print_level
from .readscube import read_scube
from .readscube import get_image_distance, radial_profile
from ..constants import FILTER_NAMES_FITS, FILTER_COLORS, FILTER_TRANSMITTANCE

fmagr = lambda x, w, p: 10**(-0.4*x)/(w**2)*(p**2)*(2.997925e18*3631.0e-23)

def crop3D(filename, hfrac_crop=None, wfrac_crop=None, output_filename=None):
    '''
    Crops a 3D image by specified height and width fractions.

    Parameters
    ----------
    filename : str
        Path to the image file to be cropped.
    
    hfrac_crop : list of float, optional
        List containing two values that define the fractional crop limits for height, 
        where 0 is the start and 1 is the end (default is [0, 1] which means no cropping in height).
    
    wfrac_crop : list of float, optional
        List containing two values that define the fractional crop limits for width,
        where 0 is the start and 1 is the end (default is [0, 1] which means no cropping in width).
    
    output_filename : str, optional
        Path to save the cropped image. If None, the image is not saved (default is None).

    Returns
    -------
    np.ndarray
        The cropped image array.

    Notes
    -----
    - The function reads the image, crops it according to the fractional limits specified for height and width,
      and returns the cropped image. If output_filename is provided, the cropped image is saved to that file.
    - Fractional cropping values should be between 0 and 1, where 0 represents the beginning and 1 the full length.
    '''    
    img = plt.imread(filename)
    h, w, _ = img.shape
    hfrac_crop = [0, 1] if hfrac_crop is None else hfrac_crop
    wfrac_crop = [0, 1] if wfrac_crop is None else wfrac_crop
    h = [int(hfrac_crop[0]*h), int(hfrac_crop[1]*h - 1)] 
    w = [int(wfrac_crop[0]*w), int(wfrac_crop[1]*w - 1)] 
    crop_img = img[h[0]:h[1], w[00]:w[1], :]
    if output_filename is not None:
        plt.imsave(output_filename, crop_img)
    return crop_img

class scube_plots():
    '''
    Class for creating various plots from SCUBE data (spectral cubes).

    This class provides methods to generate different types of plots, including images of flux, magnitude, signal-to-noise ratios, sky flux, 
    and integrated spectra. It also supports 3D flux visualization, LRGB image creation, and contour plotting.

    Parameters
    ----------
    filename : str
        The path to the FITS file containing the SCUBE data.
    
    block : bool, optional
        If True, blocks the execution of plots until they are closed (default is False).

    Attributes
    ----------
    scube_filename : str
        Path to the SCUBE file being processed.
    
    scube : read_scube
        Instance of the read_scube class, containing the data from the FITS file.
    
    block : bool
        Whether plots are blocked or not.
    
    filter_colors : np.ndarray
        Array of colors corresponding to each filter in the SCUBE.
    
    aur : float
        The golden ratio (used for figure aspect ratios).
    
    Methods
    -------
    readscube(filename)
        Reads the SCUBE data from the specified FITS file.
    
    images_plot(img_lyx, mask_yx=None, suptitle=None, output_filename=None, cmap='Spectral_r', vminmax=None)
        Plots a set of images for each filter in the data cube.
    
    images_mag_plot(output_filename=None, cmap='Spectral')
        Plots magnitude images for each filter in the data cube.
    
    images_emag_plot(output_filename=None, cmap='Spectral')
        Plots error magnitude images for each filter in the data cube.
    
    images_flux_plot(output_filename=None, cmap='Spectral_r')
        Plots flux images for each filter in the data cube in log scale.
    
    images_eflux_plot(output_filename=None, cmap='Spectral_r')
        Plots flux error images for each filter in the data cube in log scale.
    
    images_SN_plot(output_filename=None, cmap='Spectral_r')
        Plots signal-to-noise (S/N) images for each filter in the data cube.
    
    images_skyflux_plot(sky, output_filename=None, cmap='Spectral_r')
        Plots sky flux images based on the provided sky data.
    
    images_3D_plot(output_filename=None, FOV=140)
        Creates a 3D scatter plot of the flux data cube with respect to wavelength.
    
    LRGB_plot(output_filename=None, **kw_rgb)
        Creates an LRGB image from the data cube.
    
    LRGB_spec_plot(output_filename=None, i_x0=None, i_y0=None)
        Plots a spectrum with an associated LRGB image at a given pixel position.
    
    LRGB_centspec_plot(output_filename=None)
        Plots a spectrum with an associated LRGB image at the central pixel.
    
    SN_filters_plot(output_filename=None, SN_range=None, valid_mask__yx=None, bins=50)
        Plots histograms of the signal-to-noise ratio (S/N) for each filter.
    
    sky_spec_plot(sky, output_filename=None)
        Plots the sky spectrum based on the provided sky data.
    
    rings_spec_plot(output_filename=None, pa=0, ba=1, theta=None, rad_scale=1, mode='mean', sky_mask=None, rad_mask=None)
        Plots the spectrum of concentric rings from the center of the object.
    
    contour_plot(output_filename=None, contour_levels=None)
        Plots contour levels over the r-band magnitude image.
    
    int_area_spec_plot(output_filename=None, pa_deg=0, ba=1, R_pix=50)
        Plots the integrated area spectrum for a specified elliptical region.
    '''
    def __init__(self, filename, block=False):
        '''
        Initializes the scube_plots class.

        Parameters
        ----------
        filename : str
            Path to the FITS file containing the data cube.

        block : bool, optional
            Whether to block the execution of plots (default is False).
        '''        
        self.readscube(filename)       
        self.block = block
        self.filter_colors = np.array([FILTER_COLORS[FILTER_NAMES_FITS[k]] for k in self.scube.filters])
        self.aur = 0.5*(1 + 5**0.5)

    def readscube(self, filename):
        '''
        Reads the SCUBE data from the specified FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file to read.
        '''        
        self.scube_filename = filename
        self.scube = read_scube(filename)

    def images_plot(self, img__lyx, mask__yx=None, suptitle=None, output_filename=None, cmap='Spectral_r', vminmax=None):
        '''
        Plots a set of images for each filter in the data cube.

        Parameters
        ----------
        img__lyx : np.ndarray
            3D array with image data for each filter.

        mask__yx : np.ndarray, optional
            2D mask to apply to the images (default is None).

        suptitle : str, optional
            Title to display above the figure (default is None).

        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral_r').

        vminmax : list of float, optional
            Minimum and maximum values for the colormap (default is None).
        '''        
        nl = len(self.scube.filters)
        nrows = 3
        ncols = 4
        if vminmax is None:
            vmin, vmax = np.inf, -np.inf
            for i in range(nl):
                img = img__lyx[i]
                if mask__yx is not None:
                    img = np.ma.masked_array(img, mask=mask__yx, copy=True)
                _vmin, _vmax = np.percentile(img[~np.isnan(img)], [2, 98])
                vmin = _vmin if _vmin < vmin else vmin
                vmax = _vmax if _vmax > vmax else vmax
        else:
            vmin, vmax = vminmax
        f, ax_arr = plt.subplots(nrows, ncols)
        f.set_size_inches(4, 4)
        k = 0
        for ir in range(nrows):
            for ic in range(ncols):
                img = img__lyx[k]
                if mask__yx is not None:
                    img = np.ma.masked_array(img, mask=mask__yx, copy=True)
                ax = ax_arr[ir, ic]
                ax.set_title(self.scube.filters[k], fontsize=10)
                im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())       
                k += 1
        left, bottom, width, height = 0.08, 0.05, 0.98, 0.03
        cb_ax = f.add_axes([left, bottom, width, height])
        f.colorbar(im, cax=cb_ax, location='bottom')
        f.subplots_adjust(left=0.05, right=1.05, bottom=0.1, top=0.9) #, hspace=0.2, wspace=0.15)
        if suptitle is not None:
            f.suptitle(suptitle, fontsize=10)
        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def images_mag_plot(self, output_filename=None, cmap='Spectral'):
        '''
        Plots magnitude images for each filter in the data cube.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral').
        '''        
        self.images_plot(
            img__lyx=self.scube.mag__lyx, 
            suptitle=r'AB-mag/arcsec$^2$',
            output_filename=f'{self.scube.galaxy}_imgs_mag.png' if output_filename is None else output_filename,
            cmap=cmap, vminmax=[16, 25],
        )

    def images_emag_plot(self, output_filename=None, cmap='Spectral'):
        '''
        Plots error magnitude images for each filter in the data cube.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral').
        '''        
        self.images_plot(
            img__lyx=self.scube.emag__lyx, 
            suptitle=r'AB-mag/arcsec$^2$',
            output_filename=f'{self.scube.galaxy}_imgs_emag.png' if output_filename is None else output_filename,
            cmap=cmap, vminmax=[0.05, 1],
        )

    def images_flux_plot(self, output_filename=None, cmap='Spectral_r'):
        '''
        Plots flux images for each filter in the data cube in log scale.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral_r').
        '''        
        self.images_plot(
            img__lyx=np.ma.log10(self.scube.flux__lyx) + 18,
            suptitle=r'$\log_{10}$ 10$^{18}$erg/s/$\AA$/cm$^2$',
            output_filename=f'{self.scube.galaxy}_imgs_flux.png' if output_filename is None else output_filename,
            cmap=cmap, vminmax=[-1, 2.5],
        )

    def images_eflux_plot(self, output_filename=None, cmap='Spectral_r'):
        '''
        Plots flux error images for each filter in the data cube in log scale.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral_r').
        '''
        self.images_plot(
            img__lyx=np.ma.log10(self.scube.eflux__lyx) + 18,
            suptitle=r'$\log_{10}$ 10$^{18}$erg/s/$\AA$/cm$^2$',
            output_filename=f'{self.scube.galaxy}_imgs_eflux.png' if output_filename is None else output_filename,
            cmap=cmap, vminmax=[-0.5, 0.5],
        )

    def images_SN_plot(self, output_filename=None, cmap='Spectral_r'):
        '''
        Plots signal-to-noise (S/N) images for each filter in the data cube.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral_r').
        '''        
        SN__lyx = self.scube.SN__lyx

        self.images_plot(
            img__lyx=np.log10(SN__lyx),
            suptitle=r'$\log$ Signal-to-noise',
            output_filename=f'{self.scube.galaxy}_imgs_SN.png' if output_filename is None else output_filename,
            cmap=cmap, vminmax=[0, 1],
        )

    def images_skyflux_plot(self, sky, output_filename=None, cmap='Spectral_r'):
        '''
        Plots sky flux images based on the provided sky data.

        Parameters
        ----------
        sky : dict
            Sky data dictionary returned from a sky estimation function.

        output_filename : str, optional
            Path to save the output figure (default is None).

        cmap : str, optional
            Colormap to use for the images (default is 'Spectral_r').
        '''        
        f__byx = np.log10(sky['flux__lyx'])+18
        sky_pixels__yx = sky['mask__yx']

        self.images_plot(
            img__lyx=f__byx,
            mask__yx=~sky_pixels__yx,
            suptitle=r'sky flux [erg/s/$\AA$/cm$^2$]',
            output_filename=f'{self.scube.galaxy}_imgs_skyflux.png' if output_filename is None else output_filename,
            cmap=cmap,
        )

    def images_3D_plot(self, output_filename=None, FOV=140):
        '''
        Creates a 3D scatter plot of the flux data cube with respect to wavelength.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        FOV : float, optional
            Field of view (FOV) for the 3D projection (default is 140 degrees).
        '''        
        output_filename = f'{self.scube.galaxy}_imgs_3Dflux.png' if output_filename is None else output_filename

        FOV *= u.deg
        focal_lenght = 1/np.tan(FOV/2)
        print(f'FOV: {FOV}\nfocal lenght: {focal_lenght}')

        xx, yy = np.meshgrid(range(self.scube.size), range(self.scube.size))
        
        f = plt.figure()
        ax = f.add_subplot(projection='3d')

        for i, _w in enumerate(self.scube.pivot_wave):
            sc = ax.scatter(xx, yy, c=np.ma.log10(self.scube.flux__lyx[i]) + 18, 
                            zs=_w, s=1, edgecolor='none', vmin=-1, vmax=0.5, cmap='Spectral_r')
        ax.set_zticks(self.scube.pivot_wave)
        ax.set_zticklabels(self.scube.filters, rotation=-45)
        ax.set_proj_type('persp', focal_length=focal_lenght)
        ax.set_box_aspect(aspect=(7, 1, 1))
        ax.view_init(elev=20, azim=-125, vertical_axis='y')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        crop3D(output_filename, hfrac_crop=[0.3, 0.8], wfrac_crop=None, output_filename=output_filename)
        if self.block:
            plt.show(block=True)
        plt.close(f)
       
    def LRGB_plot(self, output_filename=None, **kw_rgb):
        '''
        Creates an LRGB image from the data cube.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        **kw_rgb : dict, optional
            Additional keyword arguments for controlling the LRGB image creation.
            See ``scubes.utilities.readscube.read_scubes.lRGB_image`` for help on 
            the arguments.
        '''        
        title = kw_rgb.pop('title', None)
        output_filename = f'{self.scube.galaxy}_RGBs.png' if output_filename is None else output_filename

        _kw_rgb = dict(
            rgb=['iSDSS', 'rSDSS', 'gSDSS'], 
            rgb_f=[1, 1, 1], 
            pminmax=[5, 95], 
            Q=10, 
            stretch=5, 
            im_max=1, 
            minimum=(0, 0, 0),
        )
        _kw_rgb.update(kw_rgb)

        # RGB IMG
        rgb__yxc = self.scube.lRGB_image(**_kw_rgb)

        f, ax = plt.subplots()
        f.set_size_inches(3, 3)
        ax.imshow(rgb__yxc, origin='lower')
        ax.set_title(_kw_rgb['rgb'] if title is None else title, fontsize=8)
        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def LRGB_spec_plot(self, output_filename=None, i_x0=None, i_y0=None):
        '''
        Plots a spectrum with an associated LRGB image at a given pixel position.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        i_x0 : int, optional
            x-coordinate of the pixel (default is None).

        i_y0 : int, optional
            y-coordinate of the pixel (default is None).
        '''        
        i_x0 = self.scube.i_x0 if i_x0 is None else i_x0
        i_y0 = self.scube.i_y0 if i_y0 is None else i_y0
        
        output_filename = f'{self.scube.galaxy}_LRGB_{i_x0}_{i_y0}_spec.png' if output_filename is None else output_filename

        rgb = ['iSDSS', 'rSDSS', 'gSDSS']
        
        # data
        flux__l = self.scube.flux__lyx[:, i_y0, i_x0]
        eflux__l = self.scube.eflux__lyx[:, i_y0, i_x0]
        bands__l = self.scube.pivot_wave
        
        # plot
        nrows = 4
        ncols = 2
        f = plt.figure()
        f.set_size_inches(6*self.aur, 4)
        f.subplots_adjust(left=0.05, right=1.05, bottom=0.1, top=0.9)
        gs = GridSpec(nrows=nrows, ncols=ncols, hspace=0, wspace=0.05, figure=f)
        ax = f.add_subplot(gs[0:nrows - 1, 1])
        axf = f.add_subplot(gs[-1, 1])
        axrgb = f.add_subplot(gs[:, 0])
        
        # RGB image
        rgb__yxb = self.scube.lRGB_image(
            rgb=rgb, rgb_f=[1, 1, 1], 
            pminmax=[5, 95], Q=10, stretch=5, im_max=1, minimum=(0, 0, 0)
        )
        axrgb.imshow(rgb__yxb, origin='lower')
        axrgb.set_title('R=i G=r B=g')
        
        # filters transmittance
        axf.sharex(ax)
        for i, k in enumerate(self.scube.filters):
            lt = '-' if 'JAVA' in k or 'SDSS' in k else '--'
            x = FILTER_TRANSMITTANCE[k]['wavelength']
            y = FILTER_TRANSMITTANCE[k]['transmittance']
            axf.plot(x, y, c=self.filter_colors[i], lw=1, ls=lt, label=k)
        axf.legend(loc=(0.83, 1), frameon=False, fontsize=9)
        
        # spectrum 
        ax.set_title(f'{self.scube.galaxy} @ {self.scube.tile} ({i_x0},{i_y0})')
        ax.plot(bands__l, flux__l, ':', c='k')
        ax.scatter(bands__l, flux__l, c=self.filter_colors, s=0.5)
        ax.errorbar(x=bands__l,y=flux__l, yerr=eflux__l, c='k', lw=1, fmt='|')
        ax.plot(bands__l, flux__l, '-', c='lightgray')
        ax.scatter(bands__l, flux__l, c=self.filter_colors)
        ax.axvline(x=3727, ls='--', c='k')
        ax.axvline(x=5007, ls='--', c='k')
        ax.axvline(x=6563, ls='--', c='k')
        
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        axf.set_xlabel(r'$\lambda\ [\AA]$', fontsize=10)
        axf.set_ylabel(r'${\rm R}_\lambda\ [\%]$', fontsize=10)

        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def LRGB_centspec_plot(self, output_filename=None):
        '''
        Plots a spectrum with an associated LRGB image at the central pixel.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).
        '''        
        self.LRGB_spec_plot(
            output_filename=f'{self.scube.galaxy}_LRGB_centspec.png' if output_filename is None else output_filename, 
            i_x0=self.scube.i_x0, i_y0=self.scube.i_y0
        )

    def SN_filters_plot(self, output_filename=None, SN_range=None, valid_mask__yx=None, bins=50):
        '''
        Plots histograms of the signal-to-noise ratio (S/N) for each filter.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        SN_range : list, optional
            Range of S/N values to display (default is [0, 10]).

        valid_mask__yx : np.ndarray, optional
            Mask indicating valid pixels (default is None).

        bins : int, optional
            Number of bins to use for the histogram (default is 50).
        '''        
        output_filename = f'{self.scube.galaxy}_SN_filters.png' if output_filename is None else output_filename
        SN_range = [0, 10] if SN_range is None else SN_range
    
        flux = self.scube.flux__lyx
        eflux = self.scube.eflux__lyx
        wei = self.scube.weimask__lyx
        mask__lyx = (wei > 0) | ~(np.isfinite(flux)) | ~(np.isfinite(eflux)) | (flux == 0)
        valid_mask__yx = np.ones(flux.shape[-2:], dtype='bool') if valid_mask__yx is None else valid_mask__yx
        
        f = plt.figure()
        n_rows = 4
        n_cols = int(self.scube.n_filters/2)
        gs = GridSpec(nrows=n_rows, ncols=n_cols, hspace=0.2, wspace=0.1, figure=f)
        f.set_size_inches(6, 6)
        f.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        i_col, i_row = 0, 0
        nmax = 0
        ax_tot = f.add_subplot(gs[2:, :])
        ax_tot.set_xlabel('S/N')
        for i, filt in enumerate(self.scube.filters):
            ax = f.add_subplot(gs[i_row, i_col])
            mask__yx = mask__lyx[i] | ~valid_mask__yx
            SN__yx = np.ma.masked_array(self.scube.SN__lyx[i], mask=mask__yx)
            ax.set_title(filt, color=self.filter_colors[i], fontsize=8)
            n, xe, patches = ax.hist(
                SN__yx.compressed(), bins=bins, range=SN_range, histtype='step', 
                label=f'{mask__yx.sum()} pixels', color=self.filter_colors[i], 
                lw=0.5 if 'J0' in filt else 1.5, density=True,
            )
            _ = ax_tot.hist(
                SN__yx.compressed(), bins=bins, range=SN_range, histtype='step', 
                label=f'{mask__yx.sum()} pixels', color=self.filter_colors[i], 
                lw=0.5 if 'J0' in filt else 1.5, density=True,
            )
            #despine
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            nmax = n.max() if n.max() > nmax else nmax
            ax.set_xticks([])
            # yticks only on first axis
            if i_col:
                ax.set_yticks([])
            # axis selection
            i_col += 1
            if i_col >= n_cols:
                i_col = 0
                i_row += 1
        ax_tot.legend(fontsize=8, frameon=False)
        for ax in f.axes:
            ax.set_ylim(0, 1.125*nmax)
        f.savefig(output_filename, bbox_inches='tight')
        if self.block:
            plt.show(block=True)
        plt.close(f)        

    def sky_spec_plot(self, sky, output_filename=None):        
        '''
        Plots the sky spectrum based on the provided sky data.

        Parameters
        ----------
        sky : dict
            Sky data dictionary returned from a sky estimation function.

        output_filename : str, optional
            Path to save the output figure (default is None).
        '''        
        output_filename = f'{self.scube.galaxy}_sky_spec.png' if output_filename is None else output_filename

        sky_mean_flux__l = sky['mean__l']
        sky_median_flux__l = sky['median__l']
        sky_std_flux__l = sky['std__l']
        mask__yx = sky['mask__yx']
        sky_flux__lyx = sky['flux__lyx']
        bands__l = self.scube.pivot_wave
        nl, ny, nx = sky_flux__lyx.shape

        f = plt.figure()
        f.set_size_inches(6*self.aur, 4)
        f.subplots_adjust(left=0.05, right=1.05, bottom=0.1, top=0.9)
        gs = GridSpec(nrows=4, ncols=2, hspace=0, wspace=0.05, figure=f)
        ax = f.add_subplot(gs[:, 1])
        axmask = f.add_subplot(gs[:, 0])

        i_r = self.scube.filters.index('rSDSS')
        img__yx = np.ma.masked_array(np.log10(self.scube.flux__lyx[i_r]) + 18, mask=~mask__yx, copy=True)
        img__yx = img__yx.filled(0).astype('bool')
        axmask.imshow(img__yx.astype('int'), origin='lower', cmap='Greys', interpolation='nearest')
        ax.plot(bands__l, sky_mean_flux__l, '-', c='gray', label='mean')
        ax.plot(bands__l, sky_median_flux__l, '-', c='cyan', label='median')
        y11, y12, y13, y14 = np.percentile(sky_flux__lyx, [5, 16, 84, 95], axis=(1, 2))
        ax.fill_between(bands__l, y1=y11.data, y2=y14.data, fc='lightgray')
        ax.fill_between(bands__l, y1=y12.data, y2=y13.data, fc='gray', alpha=0.3)
        ax.axhline(y=0, color='k', lw=0.5, ls='--')
        ax.errorbar(x=bands__l,y=sky_mean_flux__l, yerr=sky_std_flux__l, c='k', lw=1, fmt='|')
        ax.scatter(bands__l, sky_mean_flux__l, c=self.filter_colors, s=20, label='')
        ax.axvline(x=3727, ls='--', c='k')
        ax.axvline(x=5007, ls='--', c='k')
        ax.axvline(x=6563, ls='--', c='k')
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        ax.set_title('sky spectrum')
        
        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def rings_spec_plot(self, output_filename=None, pa=0, ba=1, theta=None, object_HLR=10, rad_scale=1, mode='mean', sky_mask=None, rad_mask=None):
        '''
        Plots the spectrum of concentric rings from the center of the object.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        pa : float, optional
            Position angle in radians (default is 0).

        ba : float, optional
            Axis ratio (default is 1).

        theta : float, optional
            Ellipse rotation angle in degrees (default is None).

        object_HLR : float, optional
            Determines the object size to create the radial bins. The function
            will create 10 pixels bins up to ``5*object_HLR``. (default is 10)

        rad_scale : float, optional
            Scaling factor for radial distances (default is 1).

        mode : str, optional
            Mode for calculating the profile ('mean' or 'median', default is 'mean').

        sky_mask : np.ndarray, optional
            Mask for the sky pixels (default is None).

        rad_mask : np.ndarray, optional
            Radial mask (default is None).
        '''        
        output_filename = f'{self.scube.galaxy}_rings_spec.png' if output_filename is None else output_filename
        from matplotlib.patches import Ellipse
        center = np.array([self.scube.x0, self.scube.y0])
        theta = pa*180/np.pi if theta is None else theta

        i_r = self.scube.filters.index('rSDSS')
        w = self.scube.pivot_wave[i_r]
        p = self.scube.pixscale
        
        rmax = int(5*self.scube.primary_header.get('SIZE_ML', object_HLR))
        step = 10
        bins__r = np.arange(0, rmax + step, step)
        bins_center__r = 0.5*(bins__r[:-1] + bins__r[1:])
    
        colors = plt.colormaps['Spectral'](bins__r/(rmax + step))
        
        flux__lr = radial_profile(
            prop=self.scube.flux__lyx, 
            x0=self.scube.x0, y0=self.scube.y0, 
            pa=pa, ba=ba,
            rad_scale=rad_scale,
            bin_r=bins__r,
            mask=rad_mask,
            mode=mode,
        )
        f = plt.figure()
        f.set_size_inches(6*self.aur, 4)
        f.subplots_adjust(left=0.05, right=1.05, bottom=0.1, top=0.9)
        gs = GridSpec(nrows=4, ncols=2, hspace=0, wspace=0.05, figure=f)
        ax = f.add_subplot(gs[:, 1])
        aximg = f.add_subplot(gs[:, 0])
        x = np.log10(self.scube.flux__lyx[i_r]) + 18
        mask__yx = x < (np.log10(fmagr(25, w, p)) + 18)
        if sky_mask is not None:
            mask__yx = sky_mask
        img__yx = np.ma.masked_array(x, mask=mask__yx, copy=True)
        aximg.imshow(img__yx.filled(-0.55), origin='lower', cmap='Greys_r', vmin=-0.5, vmax=1.5, interpolation='nearest')
        for i, color in enumerate(colors[1:]):
            ax.plot(self.scube.pivot_wave, flux__lr[:, i], 'o-', c=color)
            height = 2*bins_center__r[i]*ba
            width = 2*bins_center__r[i]
            e = Ellipse(center, height=height, width=width, angle=theta, fill=False, color=color, lw=1, ls='dotted')
            aximg.add_artist(e)
        ax.set_title(r'rings with <10pix> up to 5 R$_{50}$')
        ax.axhline(y=fmagr(23, w, p), ls='--', c='k', lw=0.4)
        ax.axhline(y=fmagr(24, w, p), ls='--', c='k', lw=0.4)
        ax.axhline(y=fmagr(25, w, p), ls='--', c='k', lw=0.4)
        ax.axvline(x=3727, ls='--', c='k')
        ax.axvline(x=5007, ls='--', c='k')
        ax.axvline(x=6563, ls='--', c='k')
        ax.set_ylim(1e-20)
        ax.set_yscale('log')
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        f.savefig(output_filename, bbox_inches='tight', dpi=300)
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def contour_plot(self, output_filename=None, contour_levels=None):
        '''
        Plots contour levels over the r-band magnitude image.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        contour_levels : list, optional
            List of contour levels to plot (default is [21, 23, 24]).
        '''        
        output_filename = f'{self.scube.galaxy}_contours.png' if output_filename is None else output_filename
        contour_levels = [21, 23, 24] if contour_levels is None else contour_levels

        i_lambda = self.scube.filters.index('rSDSS')
        image__yx = self.scube.mag__lyx[i_lambda]
        
        f, ax = plt.subplots()
        f.set_size_inches(3, 3)
        im = ax.imshow(image__yx, cmap='Spectral_r', origin='lower', vmin=16, vmax=25, interpolation='nearest')
        ax.contour(image__yx, levels=contour_levels, colors=['k', 'gray', 'lightgray'])
        plt.colorbar(im, ax=ax)
        f.savefig(output_filename, bbox_inches='tight')
        if self.block:
            plt.show(block=True)
        plt.close(f)

    def int_area_spec_plot(self, output_filename=None, pa_deg=0, ba=1, R_pix=50):
        '''
        Plots the integrated area spectrum for a specified elliptical region.

        Parameters
        ----------
        output_filename : str, optional
            Path to save the output figure (default is None).

        pa_deg : float, optional
            Position angle in degrees (default is 0).

        ba : float, optional
            Axis ratio (default is 1).

        R_pix : int, optional
            Radius in pixels for the integration area (default is 50).
        '''        
        output_filename = f'{self.scube.galaxy}_intarea_spec.png' if output_filename is None else output_filename
        
        pa_deg *= u.deg
        pa_rad = pa_deg.to('rad')

        if not (pa_deg == 0 and ba == 1):
            elliptical_pixel_distance__yx = get_image_distance(
                shape=self.scube.weimask__yx.shape, 
                x0=self.scube.i_x0, y0=self.scube.i_y0, 
                pa=pa_rad.value, ba=ba
            )
        else:
            elliptical_pixel_distance__yx = self.scube.pixel_distance__yx

        mask__yx = elliptical_pixel_distance__yx > R_pix
        __lyx = (self.scube.n_filters, self.scube.n_y, self.scube.n_x)
        mask__lyx = np.broadcast_to(mask__yx, __lyx)
        integrated_flux__lyx = np.ma.masked_array(self.scube.flux__lyx, mask=mask__lyx, copy=True)
        integrated_eflux__lyx = np.ma.masked_array(self.scube.eflux__lyx, mask=mask__lyx, copy=True)
        bands__l = self.scube.pivot_wave
        flux__l = integrated_flux__lyx.sum(axis=(1,2))
        eflux__l = (integrated_eflux__lyx**2).sum(axis=(1,2))/(bands__l.size)**2

        f = plt.figure()
        f.set_size_inches(12, 3)
        f.subplots_adjust(left=0, right=0.9)
        gs = GridSpec(nrows=1, ncols=3, wspace=0.2, figure=f)
        ax = f.add_subplot(gs[1:])
        axmask = f.add_subplot(gs[0])
        i_r = self.scube.filters.index('rSDSS')
        img__yx = np.ma.masked_array(self.scube.mag__lyx[i_r], mask=mask__yx, copy=True)
        axmask.imshow(img__yx, origin='lower', cmap='Spectral_r', vmin=16, vmax=25, interpolation='nearest')
        axmask.imshow(self.scube.mag__lyx[i_r], origin='lower', cmap='Spectral_r', alpha=0.2, vmin=16, vmax=25, interpolation='nearest')
        ax.plot(bands__l, flux__l, '-', c='lightgray')
        ax.errorbar(x=bands__l,y=flux__l, yerr=eflux__l, c='k', lw=1, fmt='|')
        ax.scatter(bands__l, flux__l, c=self.filter_colors, s=20, label='')
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        ax.set_title('int. area. spectrum')
        f.savefig(output_filename, bbox_inches='tight')
        if self.block:
            plt.show(block=True)
        plt.close(f)

def plot_mask(detection_image, lupton_rgb, masked_ddata, resulting_mask, sewregions, daoregions=None, save_fig=False, prefix_filename=None, fig=None):
    '''
    Plot a mosaic showing various images and regions related to source detection and masking.

    Parameters
    ----------
    detection_image : str
        The path to the FITS file containing the detection image.

    lupton_rgb : array-like
        The RGB image array used for plotting.

    masked_ddata : array-like
        The masked detection image data.

    resulting_mask : array-like
        The resulting mask array.

    sewregions : list of CirclePixelRegion
        List of regions around sources detected by SExtractor.

    daoregions : list of CirclePixelRegion, optional
        List of regions around sources detected by DAOStarFinder. Defaults to None.

    save_fig : bool, optional
        If True, save the figure as an image file. Defaults to False.

    prefix_filename : str, optional
        The prefix for the saved figure filename. Defaults to 'OBJECT'.
        
    fig : matplotlib.figure.Figure, optional
        The existing figure to use. If None, a new figure will be created. Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure if save_fig is False, otherwise None.
    '''    
    prefix_filename = 'OBJECT' if prefix_filename is None else prefix_filename
    
    dhdu = fits.open(detection_image)
    ddata = dhdu[1].data
    dheader = dhdu[1].header
    wcs = WCS(dheader)
    # FIGURE
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.ion()
    if fig is None:
        fig = plt.figure()
    # AX1
    ax1 = plt.subplot(221, projection=wcs)

    ax1.imshow(lupton_rgb, origin='lower')
    # r_circ.plot(color='y', lw=1.5)
    for sregion in sewregions:
        sregion.plot(ax=ax1, color='g')
    ax1.set_title('RGB')
    # AX2
    ax2 = plt.subplot(222, projection=wcs)
    ax2.imshow(ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
    # r_circ.plot(color='y', lw=1.5)
    for n, sregion in enumerate(sewregions):
        sregion.plot(ax=ax2, color='g')
        ax2.annotate(repr(n), (sregion.center.x, sregion.center.y), color='green')
    if daoregions is not None:
        for dregion in daoregions:
            dregion.plot(ax=ax2, color='m')
    ax2.set_title('Detection')
    # AX3
    ax3 = plt.subplot(223, projection=wcs)
    stars_mask = np.ones(ddata.shape)
    for n, sregion in enumerate(sewregions):
        sregion.plot(ax=ax3, color='g')
    ax3.imshow(masked_ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
    # r_circ.plot(color='y', lw=1.5)
    ax3.set_title('Masked')
    # AX4
    ax4 = plt.subplot(224, projection=wcs)
    ax4.imshow(resulting_mask, cmap='Greys_r', origin='lower')
    ax4.set_title('Mask')
    fig.subplots_adjust(wspace=.05, hspace=.2)
    for ax in [ax1, ax2, ax3, ax4]:
        if daoregions is not None:
            for dregions in daoregions:
                dregions.plot(ax=ax, color='m')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
    if save_fig:
        fig_filename = f'{prefix_filename}_maskMosaic.png'
        print_level(f'Saving fig to {fig_filename}')
        fig.savefig(fig_filename, format='png', dpi=180)
        plt.close(fig)
        fig = None
    return fig     