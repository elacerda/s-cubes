import sys
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from dataclasses import dataclass
import astropy.constants as const
from photutils import DAOStarFinder
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from os.path import join, exists, isfile
from regions import PixCoord, CirclePixelRegion
from scipy.interpolate import RectBivariateSpline
from astropy.wcs.utils import skycoord_to_pixel as sky2pix

from .control import control
from .utilities.sextractor import run_sex
from .utilities.headers import get_keys, get_author, get_key
from .utilities.constants import WAVE_EFF, NAMES_CORRESPONDENT
from .utilities.io import print_level, grep
from .utilities.splusdata import connect_splus_cloud, download_splus_stamps, download_splus_detection_image, download_splus_lupton_rgb

@dataclass
class _galaxy:
    ra: u.quantity.Quantity
    dec: u.quantity.Quantity
    name: str
    redshift: float

    def skycoord(self, frame='icrs'):
        return SkyCoord(ra=self.ra, dec=self.dec, frame=frame)
    
class SCubes:
    def __init__(self, args, program_name='SCUBES'):
        self.control = control(args, program_name=program_name)
        self.program_name = program_name
        self.galaxy = self._galaxy()
        self.galaxy.coords = self.galaxy.skycoord()
        self.conn = connect_splus_cloud(username=None, password=None)
        #self.conn = None
        self._init_spectra()

    def _init_spectra(self):
        self.wl__b = np.array([WAVE_EFF[b] for b in self.control.bands])*u.Angstrom
        self.flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
        self.fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        self.flam__b = None
        self.fnu__b = None
        self.headers__b = None

    def _check_errors(self):
        return all([exists(_) for _ in self.wimages])

    def _get_headers_list(self, images=None, ext=1):
        images = self.images if images is None else images
        return [fits.getheader(img, ext=ext) for img in images]

    def _get_data_spectra(self, images=None, ext=1):
        images = self.images if images is None else images
        return np.array([fits.getdata(img, ext=ext) for img in images])

    def _header_key_spectra(self, key):
        values = []
        for i, h in enumerate(self.headers__b):
            author = get_author(h)
            k = get_key(key, author)
            values.append(h[k])
        return np.array(values)

    def _m0(self):
        return self._header_key_spectra('MAGZP')
    
    def _gain(self):
        return self._header_key_spectra('GAIN')
    
    def _effexptime(self):
        return self._header_key_spectra('EFFTIME')
    
    def _galaxy(self):
        c = self.control
        return _galaxy(ra=c.ra, dec=c.dec, name=c.galaxy, redshift=c.specz)
    
    def check_zero_points(self):
        has_magzp = True
        for h in self.headers__b:
            mzp_key = get_key('MAGZP', get_author(h))
            if h.get(mzp_key, None) is None:
                has_magzp = False
                break
        return has_magzp
                    
    def get_stamps(self):
        gal = self.galaxy
        ctrl = self.control
        output_dir = ctrl.output_dir
        self.stamps = download_splus_stamps(
            self.conn, gal.ra, gal.dec, ctrl.size, ctrl.tile, gal.name, 
            output_dir=output_dir, 
            download_weight=True, 
            overwrite=ctrl.force
        )
        # SHORTCUTS
        self.images = [img for img in self.stamps if 'swp.' in img]
        self.wimages = [img for img in self.stamps if 'swpweight.' in img]
        # UPDATE IMAGES HEADER - WILL ADD TILE INFO AND WCS TO THE HEADERS
        for img in self.images:
            with fits.open(img, 'update') as f:
                w = WCS(f[1].header)
                f[1].header['OBJECT'] = gal.name
                f[1].header['TILE'] = ctrl.tile
                f[1].header.update(w.to_header())
        self.headers__b = self._get_headers_list(self.images, ext=1)
    
    def get_detection_image(self):
        gal = self.galaxy
        ctrl = self.control
        output_dir = ctrl.output_dir
        self.detection_image = download_splus_detection_image(
            self.conn, gal.ra, gal.dec, ctrl.size, ctrl.tile, gal.name, 
            output_dir=output_dir, 
            overwrite=ctrl.force
        )
        # UPDATE DETECTION IMAGE HEADER
        with fits.open(self.detection_image, 'update') as f:
            # UPDATE HEADER WCS
            w = WCS(f[1].header)
            f[1].header.update(w.to_header())
            # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
            author = get_author(f[1].header)
            if author == 'unknown':
                print_level('Writting header key AUTHOR to detection image', 1, ctrl.verbose)
                author = get_author(self.headers__b[0])
                f[1].header.set('AUTHOR', value=author, comment='Who ran the software')

    def get_lupton_rgb(self):
        gal = self.galaxy
        ctrl = self.control
        output_dir = ctrl.output_dir
        self.lupton_rgb = download_splus_lupton_rgb(
            self.conn, gal.ra, gal.dec, ctrl.size, ctrl.tile, gal.name, 
            output_dir=output_dir, 
            overwrite=ctrl.force
        )

    def get_zero_points_correction(self):
        """ Get corrections of zero points for location in the field. """
        ctrl = self.control
        x0, x1, nbins = 0, 9200, 32
        xgrid = np.linspace(x0, x1, nbins + 1)
        zpcorr = {}
        print_level('Getting ZP corrections for the S-PLUS bands...')
        for band in ctrl.bands:
            corrfile = join(ctrl.zpcorr_dir, 'SPLUS_' + band + '_offsets_grid.npy')
            corr = np.load(corrfile)
            zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)
        self.zpcorr = zpcorr

    def get_zero_points(self):
        ctrl = self.control
        zp_table = ctrl.zp_table
        print_level(f'Reading ZPs table: {zp_table}')
        zptab = grep(zp_table, ctrl.tile)
        zpt = pd.read_csv(zp_table)
        cols = [col.replace('ZP_', '') if 'ZP_' in col else col for col in zpt.columns]
        zpt.columns = cols
        # New class properties
        self.zptab = zpt[zpt['Field'] == ctrl.tile]
        self.get_zero_points_correction()

    def add_magzp_headers(self):
        ctrl = self.control
        gal = self.galaxy
        self.get_zero_points()
        print_level('Calibrating stamps...')
        headers = []
        for img, h in zip(self.images, self.headers__b):
            #h = fits.getheader(img, ext=1)
            h['TILE'] = ctrl.tile
            filtername = h['FILTER']
            zp = float(self.zptab[NAMES_CORRESPONDENT[filtername]].item())
            x0 = h['X0TILE']
            y0 = h['Y0TILE']
            zp += round(self.zpcorr[filtername](x0, y0)[0][0], 5)
            fits.setval(img, 'MAGZP', value=zp, comment='Magnitude zero point', ext=1)
            headers.append(fits.getheader(img, ext=1))
            print_level(f'add_magzp_headers: {img}: MAGZP={zp}', level=2, verbose=ctrl.verbose)
        # updates the list of headers with the new values
        self.headers__b = headers
         
    def calibrate_stamps(self):
        if not self.check_zero_points():
            self.add_magzp_headers()

    def get_main_circle(self):
        ctrl = self.control
        dhdu = fits.open(self.detection_image)
        ddata = dhdu[1].data
        wcs = WCS(dhdu[1].header)
        cen_coords = sky2pix(self.galaxy.coords, wcs)
        ix, iy = np.meshgrid(np.arange(ddata.shape[0]), np.arange(ddata.shape[1]))
        distance = np.sqrt((ix - cen_coords[0])**2 + (iy - cen_coords[1])**2)
        expand = True
        iteration = 1
        size = ctrl.size
        angsize = ctrl.angsize/0.55
        while expand:
            print_level(f'Iteration {iteration} - angsize: {angsize}', 1, ctrl.verbose)
            inner_mask = distance <= angsize
            disk_mask = (distance > angsize) & (distance <= angsize + 5)
            outer_mask = distance > angsize + 5
            p_inner = np.percentile(ddata[inner_mask], [16, 50, 84])
            p_disk = np.percentile(ddata[disk_mask], [16, 50, 84])
            p_outer = np.percentile(ddata[outer_mask], [16, 50, 84])
            print_level(f'Inner [16, 50, 84]: {p_inner}', 2, ctrl.verbose)
            print_level(f'Disk [16, 50, 84]: {p_inner}', 2, ctrl.verbose)
            print_level(f'Outer [16, 50, 84]: {p_inner}', 2, ctrl.verbose)
            plt.ioff()
            ax1 = plt.subplot(111, projection=wcs)
            ax1.imshow(ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
            r_circ = CirclePixelRegion(center=PixCoord(cen_coords[0], cen_coords[1]), radius=angsize)
            r_circ.plot(color='y', lw=1.5, ax=ax1, label=f'{angsize:.1f} pix')
            o_r_circ = CirclePixelRegion(center=PixCoord(cen_coords[0], cen_coords[1]), radius=angsize + 5)
            o_r_circ.plot(color='g', lw=1.5, ax=ax1, label=f'{angsize + 5:.1f} pix')
            ax1.set_title('RGB')
            ax1.set_xlabel('RA')
            ax1.set_xlabel('DEC')
            ax1.legend(loc='upper left')
            if p_disk[1] <= (p_outer[1] + (p_outer[1] - p_outer[0])):
                fig_filename = f'{ctrl.prefix_filename}_defCircle.png'
                print_level(f'Saving fig after finishing iteration {iteration}: {fig_filename}', 1, ctrl.verbose)
                plt.savefig(join(ctrl.output_dir, fig_filename), format='png', dpi=180)
                plt.close()
                expand = False
            else:
                angsize += 5
                print_level(f'Current angsize: {angsize} - size/2: {size/2}', 1, ctrl.verbose)
                if angsize >= (size/2):
                    plt.show()
                    raise ValueError(f'Iteration stopped. Angsize {angsize} bigger than size {size/2}')
                iteration += 1
        dmask = np.zeros(ddata.shape)
        dmask[distance > angsize] = 1
        return r_circ, dmask
    
    def run_sex(self):
        ctrl = self.control
        dimg = self.detection_image
        h = fits.getheader(dimg, ext=1)

        params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
                  "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO", "FWHM_IMAGE",
                  "CLASS_STAR"]

        filter_name =join(ctrl.data_dir, 'sex_data/tophat_3.0_3x3.conv')
        starnnw_name = join(ctrl.data_dir, 'sex_data/default.nnw')
        #checkimg_name = join(ctrl.output_dir, dimg.replace('detection', 'segmentation'))
        #output_name = join(ctrl.output_dir, dimg.replace('detection', 'sexcat'))
        checkimg_name = dimg.replace('detection', 'segmentation')
        output_name = dimg.replace('detection', 'sexcat')
        # configuration for SExtractor photometry
        config = {
            'DETECT_TYPE': 'CCD',
            'DETECT_MINAREA': 4,
            'DETECT_THRESH': ctrl.detect_thresh,
            'ANALYSIS_THRESH': 3.0,
            'FILTER': 'Y',
            'FILTER_NAME': filter_name,
            'DEBLEND_NTHRESH': 64,
            'DEBLEND_MINCONT': 0.0002,
            'CLEAN': 'Y',
            'CLEAN_PARAM': 1.0,
            'MASK_TYPE': 'CORRECT',
            'PHOT_APERTURES': 5.45454545,
            'PHOT_AUTOPARAMS': '3.0,1.82',
            'PHOT_PETROPARAMS': '2.0,2.73',
            'PHOT_FLUXFRAC': '0.2,0.5,0.7,0.9',
            'SATUR_LEVEL': ctrl.satur_level,
            'MAG_ZEROPOINT': 20,
            'MAG_GAMMA': 4.0,
            'GAIN': h.get(get_key('GAIN', get_author(h))),
            'PIXEL_SCALE': 0.55,
            'SEEING_FWHM': h.get(get_key('PSFFWHM', get_author(h))),
            'STARNNW_NAME': starnnw_name,
            'BACK_SIZE': ctrl.back_size,
            'BACK_FILTERSIZE': 7,
            'BACKPHOTO_TYPE': 'LOCAL',
            'BACKPHOTO_THICK': 48,
            'CHECKIMAGE_TYPE': 'SEGMENTATION',
            'CHECKIMAGE_NAME': checkimg_name,
            'NTHREADS': '2',
        }

        sewcat = run_sex(
            sex_path=ctrl.sextractor, 
            detection_fits=dimg, 
            input_config=config, 
            output_params=params, 
            work_dir=ctrl.output_dir, 
            output_file=output_name, 
            verbose=ctrl.verbose
        )

        print_level(f'Using CLASS_STAR > {ctrl.class_star:.2f} star/galaxy separator...', 1, ctrl.verbose)
        sewpos = np.transpose((sewcat['table']['X_IMAGE'], sewcat['table']['Y_IMAGE']))
        radius = 3.0 * (sewcat['table']['FWHM_IMAGE'] / 0.55)
        sidelim = 80
        # CHECK NAXIS1,2 or 2,1
        shape = (h.get('NAXIS2'), h.get('NAXIS1'))
        mask = sewcat['table']['CLASS_STAR'] > ctrl.class_star
        mask &= sewcat['table']['X_IMAGE'] > sidelim
        mask &= sewcat['table']['X_IMAGE'] < (shape[0] - sidelim)
        mask &= sewcat['table']['Y_IMAGE'] > sidelim
        mask &= sewcat['table']['Y_IMAGE'] < (shape[0] - sidelim)
        mask &= sewcat['table']['FWHM_IMAGE'] > 0
        sewregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(sewpos[mask], radius[mask])]
        
        return sewregions

    def DAOfinder(self, data):
        "calculate photometry using DAOfinder"
        # DETECT TOO MUCH HII REGIONS
        mean, median, std = 0, 0, 0.5
        print_level(('mean', 'median', 'std'))
        print_level((mean, median, std))
        print_level('Running DAOfinder...')
        daofind = DAOStarFinder(fwhm=4.0, sharplo=0.2, sharphi=0.9, roundlo=-0.5, roundhi=0.5, threshold=5. * std)
        sources = daofind(data)
        return sources
    
    def update_masks(self, sewregions, detection_mask, unmask_stars=None):
        ctrl = self.control
        unmask_stars = [] if unmask_stars is None else unmask_stars
        ddata = fits.getdata(self.detection_image, ext=1)
        stars_mask = np.ones(ddata.shape)
        for n, sregion in enumerate(sewregions):
            if n not in unmask_stars:
                mask = sregion.to_mask()
                if (min(mask.bbox.extent) < 0) or (max(mask.bbox.extent) > ctrl.size):
                    print_level(f'Region is out of range for extent {mask.bbox.extent}')
                else:
                    _slices = (slice(mask.bbox.iymin, mask.bbox.iymax), slice(mask.bbox.ixmin, mask.bbox.ixmax))
                    print_level(f'{mask.bbox.extent} min: {min(mask.bbox.extent)} {_slices}', 2, ctrl.verbose)
                    stars_mask[_slices] *= 1 - mask.data
        stars_mask = np.where(stars_mask == 1, 0, 2)
        resulting_mask = detection_mask + stars_mask
        masked_ddata = np.where(resulting_mask > 0, 0, ddata)
        return masked_ddata, resulting_mask

    def plot_mask(self, masked_ddata, resulting_mask, r_circ, sewregions, daoregions=None, save_fig=False):
        ctrl = self.control
        dhdu = fits.open(self.detection_image)
        ddata = dhdu[1].data
        dheader = dhdu[1].header
        wcs = WCS(dheader)
        # FIGURE
        plt.rcParams['figure.figsize'] = (12, 10)
        plt.ion()
        fig = plt.figure()
        # ax1
        ax1 = plt.subplot(221, projection=wcs)
        self.get_lupton_rgb()
        ax1.imshow(self.lupton_rgb, origin='lower')
        r_circ.plot(color='y', lw=1.5)
        for sregion in sewregions:
            sregion.plot(ax=ax1, color='g')
        ax1.set_title('RGB')
        # ax2
        ax2 = plt.subplot(222, projection=wcs)
        ax2.imshow(ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
        r_circ.plot(color='y', lw=1.5)
        for n, sregion in enumerate(sewregions):
            sregion.plot(ax=ax2, color='g')
            ax2.annotate(repr(n), (sregion.center.x, sregion.center.y), color='green')
        if daoregions is not None:
            for dregion in daoregions:
                dregion.plot(ax=ax2, color='m')
        ax2.set_title('Detection')
        # ax3
        ax3 = plt.subplot(223, projection=wcs)
        stars_mask = np.ones(ddata.shape)
        for n, sregion in enumerate(sewregions):
            sregion.plot(ax=ax3, color='g')
        ax3.imshow(masked_ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
        r_circ.plot(color='y', lw=1.5)
        ax3.set_title('Masked')
        # ax4
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
            fig_filename = join(ctrl.output_dir, f'{ctrl.prefix_filename}_maskMosaic.png')
            print_level(f'Saving fig to {fig_filename}')
            fig.savefig(fig_filename, format='png', dpi=180)
            plt.close(fig)
            fig = None
        return fig     
   
    def _calc_masks(self, save_fig=False, run_DAOfinder=False, unmask_stars=None):
        print_level('Calculating mask...')
        r_circ, detection_mask = self.get_main_circle()
        print_level('Running SExtractor to get photometry...')
        sewregions = self.run_sex()
        daoregions = None
        if run_DAOfinder:
            daocat = self.DAOfinder(data=fits.getdata(self.detection_image, ext=1))
            daopos = np.transpose((daocat['xcentroid'], daocat['ycentroid']))
            daorad = 4*(abs(daocat['sharpness']) + abs(daocat['roundness1']) + abs(daocat['roundness2']))
            daoregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(daopos, daorad)]
        masked_ddata, resulting_mask = self.update_masks(sewregions=sewregions, detection_mask=detection_mask, unmask_stars=unmask_stars)
        fig = self.plot_mask(masked_ddata, resulting_mask, r_circ, sewregions, daoregions=daoregions, save_fig=save_fig)
        return resulting_mask, fig
    
    def _create_mask_hdu(self, resulting_mask, save_mask=False):
        ctrl = self.control
        dhdu = fits.open(self.detection_image)
        mhdul = dhdu.copy()
        mhdul[1].data = resulting_mask
        if save_mask:
            mhdul[1].header['IMGTYPE'] = ('MASK', 'boolean mask')
            del mhdul[1].header['EXPTIME']
            #del mhdul[1].header['FILTER']
            del mhdul[1].header[get_key('GAIN', get_author(mhdul[1].header))]
            #del mhdul[1].header[get_key('PSFFWHM', get_author(mhdul[1].header))]
            mask_filename = join(ctrl.output_dir, f'{ctrl.prefix_filename}_mask.fits')
            print_level(f'Saving mask to {mask_filename}')
            mhdul.writeto(mask_filename, overwrite=True)
        return mhdul

    def create_mask_hdu(self):
        ctrl = self.control
        mask_filename = f'{ctrl.prefix_filename}_mask.fits'
        if isfile(mask_filename):
            resulting_mask = fits.open(mask_filename)
        else:           
            resulting_mask, fig = self._calc_masks()
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
                    fig_filename = join(ctrl.output_dir, f'{ctrl.prefix_filename}_maskMosaic.png')
                    print_level(f'Saving fig to {fig_filename}', 1, ctrl.verbose)
                    fig.savefig(fig_filename, format='png', dpi=180)
                    plt.close(fig)
                elif in_opt == 'q':
                    Warning('Exiting!')
                    sys.exit(1)
                else:
                    raise IOError('Option %s not recognized' % in_opt)
                if len(unmask_stars) or in_opt == 'r':
                    resulting_mask, fig = self._calc_masks(unmask_stars=unmask_stars)
        return self._create_mask_hdu(resulting_mask, save_mask=True)

    def stamp_WCS_to_cube_header(self, header):
        img = self.images[0]
        w = WCS(header)
        w = WCS(fits.getheader(img, 1))
        nw = WCS(naxis=3)
        nw.wcs.cdelt[:2] = w.wcs.cdelt
        nw.wcs.crval[:2] = w.wcs.crval
        nw.wcs.crpix[:2] = w.wcs.crpix
        nw.wcs.ctype[0] = w.wcs.ctype[0]
        nw.wcs.ctype[1] = w.wcs.ctype[1]
        try:
            nw.wcs.pc[:2, :2] = w.wcs.pc
        except:
            pass
        return nw.to_header()

    def create_metadata_hdu(self, keys=None):
        tab = []
        names = ['FILTER', 'WAVE_EFF', 'EXPTIME']
        tab.append(self.control.bands)
        tab.append(self.wl__b)
        tab.append(self.effexptime__b)

        # create metadata arrays from headers keys
        keys = ['GAIN', 'PSFFWHM', 'DATE-OBS'] if keys is None else keys
        for key in keys:
            fcount = 0
            list_values = []
            for h in self.headers__b:
                k = get_key(key, get_author(h))
                val = h.get(k, None)
                if val is None:
                    raise ValueError(f'Missing value: {key}: {val}')
                else:
                    fcount += 1
                    list_values.append(val)
            if fcount != len(self.headers__b):
                raise ValueError(f'Missing header key {key}')
            tab.append(list_values)
            names.append(key)
        # Bin Table HDU
        meta_tab = Table(tab, names=names)
        meta_hdu = fits.BinTableHDU(meta_tab)
        return meta_hdu

    def spectra(self, flam_scale=None):
        flam_scale = 1e-19 if flam_scale is None else flam_scale
        _c = const.c
        scale = (1/flam_scale)
    
        self.m0__b = self._m0()
        self.gain__b = self._gain()
        self.effexptime__b = self._effexptime()
        self.data__byx = self._get_data_spectra(self.images, 1)   
        self.f0__b = np.power(10, -0.4*(48.6 + self.m0__b))
        self.fnu__byx = self.data__byx*self.f0__b[:, None, None]*self.fnu_unit
        self.flam__byx = scale*(self.fnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

        if self._check_errors():
            weidata = self._get_data_spectra(self.wimages, 1)
            dataclip = np.clip(self.data__byx, 0, np.infty)
            gain__byx = self.gain__b[:, None, None]
            dataerr = 1/weidata + dataclip/gain__byx
            self.efnu__byx = dataerr*self.f0__b[:, None, None]*self.fnu_unit
            self.eflam__byx = scale*(self.efnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

    def create_cube(self, cube_file=None, get_mask=True, flam_scale=None):
        flam_scale = 1e19 if flam_scale is None else flam_scale
        ctrl = self.control
        cube_h = self.headers__b[0].copy()
        _cube_file =  join(ctrl.output_dir, f'{self.galaxy.name}-cube.fits')
        cube_file =_cube_file if cube_file is None else cube_file

        # CREATE SPECTRA
        self.spectra(flam_scale=flam_scale)

        # DELETE BOGUS INFO
        for _k in ['FILTER', 'MAGZP', 'NCOMBINE', 'EFFTIME', 'GAIN', 'PSFFWHM']:
            k = get_key(_k, get_author(cube_h))
            if cube_h.get(k, None) is not None:
                print_level(f'create_cube: deleting header key {k}', 2, ctrl.verbose)
                del cube_h[k]    

        # UPDATE WCS IN HEADER
        cube_h.update(self.stamp_WCS_to_cube_header(cube_h))

        # CREATE CUBE
        prim_hdu = fits.PrimaryHDU()
        flam_hdu = fits.ImageHDU(self.flam__byx, cube_h)
        flam_hdu.header['EXTNAME'] = ('DATA', 'Name of the extension')
        flam_hdu.header['SPECZ'] = (ctrl.specz, 'Spectroscopic redshift')
        flam_hdu.header['PHOTZ'] = ('', 'Photometric redshift')
        hdu_list = [prim_hdu, flam_hdu]
        if self._check_errors():
            eflam_hdu = fits.ImageHDU(self.eflam__byx, cube_h)
            eflam_hdu.header['EXTNAME'] = ('ERRORS', 'Name of the extension')
            hdu_list.append(eflam_hdu)
        for hdu in hdu_list:
            hdu.header['BSCALE'] = (flam_scale, 'Linear factor in scaling equation')
            hdu.header['BZERO'] = (0, 'Zero point in scaling equation')
            hdu.header['BUNIT'] = (f'{self.flam_unit}', 'Physical units of the array values')
        if get_mask:
            mask_hdul = self.create_mask_hdu()     # HDUList
            mask_hdu = mask_hdul[1].copy()
            mask_hdu.header['EXTNAME'] = ('MASK', 'Boolean mask of the galaxy')
            hdu_list.append(mask_hdu)
        meta_hdu = self.create_metadata_hdu()  # BinTableHDU
        meta_hdu.header['EXTNAME'] = 'METADATA'
        hdu_list.append(meta_hdu)

        # SAVE CUBE
        print_level(f'writting cube {cube_file}', 1, ctrl.verbose)
        fits.HDUList(hdu_list).writeto(cube_file, overwrite=True)
        print_level(f'Cube successfully created!')

    def make(self, get_mask=True, det_img=True, flam_scale=None):
        ctrl = self.control
        output_dir = ctrl.output_dir
        cube_filename = f'{ctrl.prefix_filename}_cube.fits'
        cube_path = join(output_dir, cube_filename)
        if exists(cube_path) and not ctrl.redo:
            raise OSError('Cube exists!')
        self.get_stamps()
        if get_mask and not det_img:
            print_level('For mask detection image is required. Overwriting det_img=True')
            det_img = True
        if det_img:
            self.get_detection_image()
        self.calibrate_stamps()
        self.create_cube(cube_path, get_mask=get_mask, flam_scale=flam_scale)