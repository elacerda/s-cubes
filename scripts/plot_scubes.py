import sys
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec    

from scubes.utilities.readscube import read_scube
from scubes.utilities.readscube import get_image_distance
from scubes.constants import FILTER_NAMES_FITS, FILTER_COLORS, FILTER_TRANSMITTANCE

class scube_plots:
    def __init__(self, filename):
        self.scube = read_scube(filename)
        self.SNAME = self.scube.galaxy

    def images_mag_plot(self, output_filename=None):
        scube = self.scube
        
        output_filename = f'{self.SNAME}_imgs_mag.png' if output_filename is None else output_filename

        filters_names = scube.metadata['FILTER']
        f__byx = scube.mag__lyx
        ef__byx = scube.emag__lyx   
        nb = len(filters_names)
        
        nrows = 2
        ncols = int(nb/nrows)

        f, ax_arr = plt.subplots(2*nrows, ncols)
        f.set_size_inches(12, 6)
        f.subplots_adjust(left=0.01, right=0.95, bottom=0.05, top=0.90, hspace=0.22, wspace=0.13)

        k = 0
        for ir in range(nrows):
            for ic in range(ncols):
                img = f__byx[k]
                eimg = ef__byx[k]

                vmin, vmax = 16, 25
                ax = ax_arr[ir*2, ic]
                ax.set_title(filters_names[k])
                im = ax.imshow(img, origin='lower', cmap='Spectral', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())

                vmin, vmax = 0, 0.5
                ax = ax_arr[ir*2 + 1, ic]
                ax.set_title(f'err {filters_names[k]}')
                im = ax.imshow(eimg, origin='lower', cmap='Spectral', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                
                k += 1
        f.suptitle(r'mag/arcsec/pix$^2$')
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def images_flux_plot(self, output_filename=None):
        scube = self.scube
        
        output_filename = f'{self.SNAME}_imgs_flux.png' if output_filename is None else output_filename

        filters_names = scube.metadata['FILTER']
        f__byx = np.ma.log10(scube.flux__lyx) + 18
        ef__byx = np.ma.log10(scube.eflux__lyx) + 18
        nb = len(filters_names)

        nrows = 2
        ncols = int(nb/nrows)
        
        f, ax_arr = plt.subplots(2*nrows, ncols)
        f.set_size_inches(12, 6)
        f.subplots_adjust(left=0.01, right=0.95, bottom=0.05, top=0.90, hspace=0.22, wspace=0.13)

        k = 0
        for ir in range(nrows):
            for ic in range(ncols):
                img = f__byx[k]
                eimg = ef__byx[k]

                vmin, vmax = np.percentile(img.compressed(), [5, 95])
                vmin, vmax = -1, 1
                ax = ax_arr[ir*2, ic]
                ax.set_title(filters_names[k])
                im = ax.imshow(img, origin='lower', cmap='Spectral', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())

                vmin, vmax = np.percentile(eimg.compressed(), [5, 95])
                vmin, vmax = -1, 1
                ax = ax_arr[ir*2 + 1, ic]
                ax.set_title(f'err {filters_names[k]}')
                im = ax.imshow(eimg, origin='lower', cmap='Spectral', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())

                k += 1
        f.suptitle(r'$\log_{10}$ 10$^{18}$erg/s/$\AA$/cm$^2$')
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def images_3D_plot(self, output_filename=None, FOV=140):
        scube = self.scube
        
        output_filename = f'{self.SNAME}_imgs_3Dflux.png' if output_filename is None else output_filename

        FOV *= u.deg
        focal_lenght = 1/np.tan(FOV/2)
        print(f'FOV: {FOV}\nfocal lenght: {focal_lenght}')

        xx, yy = np.meshgrid(range(scube.size), range(scube.size))
        
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        for i, _w in enumerate(scube.pivot_wave):
            sc = ax.scatter(xx, yy, c=np.ma.log10(scube.flux__lyx[i]) + 18, 
                            zs=_w, s=1, edgecolor='none', vmin=-1, vmax=0.5, cmap='Spectral_r')
        ax.set_zticks(scube.pivot_wave)
        ax.set_zticklabels(scube.filters, rotation=-45)
        ax.set_proj_type('persp', focal_length=focal_lenght)
        ax.set_box_aspect(aspect=(7, 1, 1))
        ax.view_init(elev=20, azim=-125, vertical_axis='y')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def RGB_plot(self, output_filename=None, **kw_rgb):
        scube = self.scube
        
        title = kw_rgb.pop('title', None)
        output_filename = f'{self.SNAME}_RGBs.png' if output_filename is None else output_filename

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
        rgb__yxc = scube.lRGB_image(**_kw_rgb)

        f, ax = plt.subplots()
        ax.imshow(rgb__yxc, origin='lower')
        ax.set_title(_kw_rgb['rgb'] if title is None else title)
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def LRGB_centspec_filters_plot(self, output_filename=None, rgb=None):
        scube = self.scube
        
        output_filename = f'{self.SNAME}_LRGB_centspec.png' if output_filename is None else output_filename

        # central coords
        i_x0, i_y0 = scube.i_x0, scube.i_y0
        spec_pix_y, spec_pix_x = i_y0, i_x0
        rgb = ['iSDSS', 'rSDSS', 'gSDSS'] if rgb is None else rgb
        
        # data
        flux__l = scube.flux__lyx[:, i_y0, i_x0]
        eflux__l = scube.eflux__lyx[:, i_y0, i_x0]lRGB_image
        bands__l = scube.pivot_wave
        
        # plot
        nrows = 4
        ncols = 2
        f = plt.figure()
        f.set_size_inches(12, 5)
        f.subplots_adjust(left=0, right=0.9)
        gs = GridSpec(nrows=nrows, ncols=ncols, hspace=0, wspace=0.03, figure=f)
        ax = f.add_subplot(gs[0:nrows - 1, 1])
        axf = f.add_subplot(gs[-1, 1])
        axrgb = f.add_subplot(gs[:, 0])
        
        # RGB image
        rgb__yxb = scube.lRGB_image(
            rgb=rgb, rgb_f=[1, 1, 1], 
            pminmax=[5, 95], Q=10, stretch=5, im_max=1, minimum=(0, 0, 0)
        )
        axrgb.imshow(rgb__yxb, origin='lower')
        axrgb.set_title(rgb)
        
        # filters transmittance
        axf.sharex(ax)
        for i, k in enumerate(scube.filters):
            lt = '-' if 'JAVA' in k or 'SDSS' in k else '--'
            x = FILTER_TRANSMITTANCE[k]['wavelength']
            y = FILTER_TRANSMITTANCE[k]['transmittance']
            axf.plot(x, y, c=self.filter_colors[i], lw=1, ls=lt, label=k)
        axf.legend(loc=(0.82, 1.15), frameon=False)
        
        # spectrum 
        ax.set_title(f'{self.SNAME} @ {scube.tile} ({spec_pix_x},{spec_pix_y})')
        ax.plot(bands__l, flux__l, ':', c='k')
        ax.scatter(bands__l, flux__l, c=self.filter_colors, s=0.5)
        ax.errorbar(x=bands__l,y=flux__l, yerr=eflux__l, c='k', lw=1, fmt='|')
        ax.plot(bands__l, flux__l, '-', c='lightgray')
        ax.scatter(bands__l, flux__l, c=self.filter_colors)
        
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        axf.set_xlabel(r'$\lambda\ [\AA]$', fontsize=10)
        axf.set_ylabel(r'${\rm R}_\lambda\ [\%]$', fontsize=10)

        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def contour_plot(self, output_filename=None, contour_levels=None):
        scube = self.scube
        output_filename = f'{self.SNAME}_contours.png' if output_filename is None else output_filename
        contour_levels = [21, 23, 24] if contour_levels is None else contour_levels

        i_lambda = scube.filters.index('rSDSS')
        image__yx = scube.mag__lyx[i_lambda]
        
        f, ax = plt.subplots()
        im = ax.imshow(image__yx, cmap='Spectral_r', origin='lower', vmin=16, vmax=25)
        ax.contour(image__yx, levels=contour_levels, colors=['k', 'gray', 'lightgray'])
        plt.colorbar(im, ax=ax)
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    def int_area_spec_plot(self, output_filename=None, pa_deg=0, ba=1, R_pix=50):
        scube = self.scube
        output_filename = f'{self.SNAME}_intarea_spec.png' if output_filename is None else output_filename
        
        pa_deg *= u.deg
        pa_rad = pa_deg.to('rad')

        if not (pa_deg == 0 and ba == 1):
            elliptical_pixel_distance__yx = get_image_distance(
                shape=scube.weimask__yx.shape, 
                x0=scube.i_x0, y0=scube.i_y0, 
                pa=pa_rad.value, ba=ba
            )
        else:
            elliptical_pixel_distance__yx = scube.pixel_distance__yx

        mask__yx = elliptical_pixel_distance__yx > R_pix
        __lyx = (scube.n_filters, scube.ny, scube.nx)
        mask__lyx = np.broadcast_to(mask__yx, __lyx)
        integrated_flux__lyx = np.ma.masked_array(scube.flux__lyx, mask=mask__lyx, copy=True)
        bands__l = scube.pivot_wave
        flux__l = integrated_flux__lyx.sum(axis=(1,2))

        f = plt.figure()
        f.set_size_inches(12, 4)
        f.subplots_adjust(left=0, right=0.9)
        gs = GridSpec(nrows=1, ncols=3, wspace=0.2, figure=f)
        ax = f.add_subplot(gs[1:])
        axmask = f.add_subplot(gs[0])
        img__yx = np.ma.masked_array(scube.mag__lyx[scube.filters.index('rSDSS')], mask=mask__yx, copy=True)
        axmask.imshow(img__yx, origin='lower', cmap='Spectral_r')
        axmask.imshow(scube.mag__lyx[scube.filters.index('rSDSS')], origin='lower', cmap='Spectral_r', alpha=0.5, vmin=16, vmax=25)
        ax.plot(bands__l, flux__l, '-', c='lightgray')
        ax.scatter(bands__l, flux__l, c=self.filter_colors, s=20, label='')
        ax.set_xlabel(r'$\lambda_{\rm pivot}\ [\AA]$', fontsize=10)
        ax.set_ylabel(r'flux $[{\rm erg}\ \AA^{-1}{\rm s}^{-1}{\rm cm}^{-2}]$', fontsize=10)
        ax.set_title('int. area. spectrum')
        f.savefig(output_filename, bbox_inches='tight')
        plt.close(f)

    @property
    def filter_colors(self):
        return np.array([FILTER_COLORS[FILTER_NAMES_FITS[k]] for k in self.scube.filters])

if __name__ == '__main__':
    splots = scube_plots(filename=sys.argv[1])
    
    ofile = f'{splots.SNAME}_imgs_mag.png'
    splots.images_mag_plot(output_filename=ofile)
    
    ofile = f'{splots.SNAME}_imgs_flux.png'
    splots.images_flux_plot(output_filename=ofile)
    
    ofile = f'{splots.SNAME}_imgs_3Dflux.png'
    splots.images_3D_plot(output_filename=ofile)
    
    kw_rgb = dict(
        rgb=['iSDSS', 'rSDSS', 'gSDSS'], 
        rgb_f=[1, 1, 1], 
        pminmax=[5, 95], 
        Q=10, 
        stretch=5, 
        im_max=1, 
        minimum=(0, 0, 0)
    )
    ofile = f'{splots.SNAME}_RGB_irg.png'
    kw_rgb['title'] = 'R=I G=R B=G'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = ['iSDSS', 'rSDSS', (0, 1, 2, 3, 4)]
    kw_rgb['title'] = 'R=I G=R B=U,J0378,395,410,430'
    ofile = f'{splots.SNAME}_RGB_ir0to4.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = ['iSDSS', 'rSDSS', 'gSDSS']
    kw_rgb['rgb_f'] = [1, 1, 2]
    kw_rgb['title'] = r'R=I G=R B=$2\times$G'
    ofile = f'{splots.SNAME}_RG2B_irg.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = [(7, 9, 11), 5, (0, 1, 2, 3, 4)]
    kw_rgb['rgb_f'] = [1, 1, 1]
    kw_rgb['pminmax'] = [1.5, 98.5]
    kw_rgb['Q'] = 3
    kw_rgb['stretch'] = 130
    kw_rgb['im_max'] = 180
    kw_rgb['minimum'] = (15, 15, 15)
    kw_rgb['title'] = 'R=R,I,Z G=G B=U,J0378,395,410,430'
    ofile = f'{splots.SNAME}_RGB_riz_g_0to4.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = ['zSDSS', 'gSDSS', 'uJAVA']
    kw_rgb['title'] = 'R=Z G=G B=U'
    ofile = f'{splots.SNAME}_RGB_zgu.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = [9, (2, 3, 4),  (0, 1)]
    kw_rgb['title'] = 'R=I G=J0395,410,430 B=U,J0378'
    ofile = f'{splots.SNAME}_RGB_i_2to4_01.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    kw_rgb['rgb'] = [8, 5,  (0, 1, 2, 3, 4)]
    kw_rgb['title'] = 'R=J0660 G=G B=U,J0378,395,410,430'
    ofile = f'{splots.SNAME}_RGB_i_2to4_01.png'
    splots.RGB_plot(output_filename=ofile, **kw_rgb)

    ofile = f'{splots.SNAME}_LRGB_centspec.png'
    splots.LRGB_centspec_filters_plot(output_filename=ofile, rgb=['iSDSS', 'rSDSS', 'gSDSS'])
    
    contour_levels = [21, 23, 24]
    ofile = f'{splots.SNAME}_contours_mag_21_23_24.png'
    splots.contour_plot(output_filename=ofile, contour_levels=contour_levels)
    
    ofile = f'{splots.SNAME}_intarea_rad50pix_spec.png'
    splots.int_area_spec_plot(output_filename=ofile, pa_deg=0, ba=1, R_pix=50)