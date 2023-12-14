import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from os import remove, makedirs
from astropy.table import Table
from dataclasses import dataclass
import astropy.constants as const
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from os.path import join, exists, isfile
from scipy.interpolate import RectBivariateSpline
#from astropy.wcs.utils import skycoord_to_pixel as sky2pix

from .control import control
from .headers import get_author, get_key
from .constants import WAVE_EFF, NAMES_CORRESPONDENT, \
    SPLUS_DEFAULT_SEXTRACTOR_CONFIG, \
    SPLUS_DEFAULT_SEXTRACTOR_PARAMS

from .utilities.io import print_level
from .utilities.plots import plot_mask
from .utilities.stats import robustStat
from .utilities.daofinder import DAOregions
from .utilities.sextractor import run_sex, SEWregions, unmask_sewregions
from .utilities.splusdata import connect_splus_cloud, detection_image_hdul, get_lupton_rgb

@dataclass
class _galaxy:
    ra: u.quantity.Quantity
    dec: u.quantity.Quantity
    name: str
    redshift: float

    def skycoord(self, frame='icrs'):
        return SkyCoord(ra=self.ra, dec=self.dec, frame=frame, unit='deg')
    
class _control(control):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = join(self.work_dir, self.galaxy)
        print_level(f'output_dir: {self.output_dir}', 2, self.verbose)
        print_level(f'prefix_filename: {self.prefix_filename}', 2, self.verbose)
        self._make_output_dir()

    def _make_output_dir(self):
        try: 
            makedirs(self.output_dir)
        except FileExistsError:
            print_level(f'{self.output_dir}: directory already exists', 2, self.verbose)    

class SCubes:
    def __init__(self, args):
        self._conn = None
        self.control = _control(args)
        self._init_galaxy()
        self._init_spectra()

    def _init_galaxy(self):
        self.galaxy = self._galaxy()
        gal = self.galaxy
        gal.ra = self.control.ra
        gal.dec = self.control.dec
        gal.name = self.control.galaxy
        gal.redshift = self.control.specz
        self.galaxy.coords = self.galaxy.skycoord()

    def _init_spectra(self):
        self.wl__b = np.array([WAVE_EFF[b] for b in self.control.bands])*u.Angstrom
        self.flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
        self.fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        self.flam__b = None
        self.fnu__b = None
        self.headers__b = None

    @property
    def conn(self):
        if self._conn is None:
            ctrl = self.control
            self._conn = connect_splus_cloud(ctrl.username, ctrl.password)
        return self._conn

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
        self.stamps = []
        for filt in tqdm(list(WAVE_EFF.keys()), desc=f'{gal.name} @ {ctrl.tile} - downloading', leave=True, position=0):
            fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{filt}_{ctrl.size}x{ctrl.size}_swp.fits.fz')
            self.stamps.append(fname)
            if not isfile(fname) or ctrl.force:
                _ = self.conn.stamp(ra=gal.ra, dec=gal.dec, size=ctrl.size, band=filt, weight=False, option=ctrl.tile, filename=fname)
            # Download_weight:
            fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{filt}_{ctrl.size}x{ctrl.size}_swpweight.fits.fz')
            self.stamps.append(fname)
            if not isfile(fname) or ctrl.force:
                _ = self.conn.stamp(ra=gal.ra, dec=gal.dec, size=ctrl.size, band=filt, weight=True, option=ctrl.tile, filename=fname)
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
        band = 'G,R,I,Z'
        self.detection_image = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{ctrl.size}x{ctrl.size}_detection.fits')
        if not isfile(self.detection_image) or ctrl.force:
            print_level(f' {gal.name} @ {ctrl.tile} - downloading detection image')
            kw = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, bands=band, option=ctrl.tile)
            hdul = detection_image_hdul(self.conn, wcs=True, **kw)
            author = get_author(hdul[1].header)

            # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
            if author == 'unknown':
                print_level('Writting header key AUTHOR to detection image', 1, ctrl.verbose)
                author = get_author(self.headers__b[0])
                hdul[1].header.set('AUTHOR', value=author, comment='Who ran the software')

            hdul.writeto(self.detection_image, overwrite=ctrl.force)
                        
    def get_lupton_rgb(self):
        gal = self.galaxy
        ctrl = self.control
        #conn = self.get_splusdata_conn()
        fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{ctrl.size}x{ctrl.size}.png')
        if not isfile(fname) or ctrl.force:
            print_level(f'{gal.name} @ {ctrl.tile} - downloading RGB image')
            kw = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, option=ctrl.tile)
            img = get_lupton_rgb(self.conn, transpose=True, **kw)
        else:
            img = Image.open(fname)
        self.lupton_rgb = img

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

    '''
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
    '''

    def run_sex(self):
        ctrl = self.control
        dimg = self.detection_image

        checkimg_name = dimg.replace('detection', 'segmentation')
        output_name = dimg.replace('detection', 'sexcat')

        # configuration for SExtractor photometry
        i = 0 if ctrl.estimate_fwhm else 1

        while i < 2:
            h = fits.getheader(dimg, ext=1)

            config = SPLUS_DEFAULT_SEXTRACTOR_CONFIG
            config.update({
                'DETECT_THRESH': ctrl.detect_thresh,
                'SATUR_LEVEL': ctrl.satur_level,
                'GAIN': h.get(get_key('GAIN', get_author(h))),
                'SEEING_FWHM': h.get(get_key('PSFFWHM', get_author(h))),  #, h.get('PSFFWHM', None)),
                'BACK_SIZE': ctrl.back_size,
                'CHECKIMAGE_NAME': checkimg_name,
            })

            sewcat = run_sex(
                sex_path=ctrl.sextractor, 
                detection_fits=dimg, 
                input_config=config, 
                output_params=SPLUS_DEFAULT_SEXTRACTOR_PARAMS, 
                work_dir=ctrl.output_dir, 
                output_file=output_name, 
                verbose=ctrl.verbose
            )

            if not i and ctrl.estimate_fwhm:
                stats = robustStat(sewcat['table']['FWHM_IMAGE']) 
                psffwhm = stats['median']*0.55
                fits.setval(self.detection_image, 'HIERARCH OAJ PRO FWHMMEAN', value=psffwhm, comment='', ext=1)
                files_to_remove = ['params.txt', 'conv.txt', 'config.txt', 'default.psf']
                for _f in files_to_remove:
                    f = join(ctrl.output_dir, _f)
                    remove(f)
            i += 1

        return SEWregions(sewcat=sewcat, class_star=ctrl.class_star, shape=(h.get('NAXIS2'), h.get('NAXIS1')), verbose=ctrl.verbose)
     
    def _calc_masks(self, save_fig=False, run_DAOfinder=False, unmask_stars=None):
        ctrl = self.control
        print_level('Calculating mask...')
        #r_circ, detection_mask = self.get_main_circle()
        print_level('Running SExtractor to get photometry...')
        sewregions = self.run_sex()
        ddata = fits.getdata(self.detection_image, ext=1)
        daoregions = None
        if run_DAOfinder:
            daoregions = DAOregions(data=ddata)
        #masked_ddata, resulting_mask = self.update_masks(sewregions=sewregions, detection_mask=detection_mask, unmask_stars=unmask_stars)
        masked_ddata, resulting_mask = unmask_sewregions(data=ddata, sewregions=sewregions, size=ctrl.size, unmask_stars=unmask_stars, verbose=ctrl.verbose)
        self.get_lupton_rgb()
        prefix_filename = join(ctrl.output_dir, ctrl.prefix_filename)
        fig = plot_mask(
            detection_image=self.detection_image, 
            lupton_rgb=self.lupton_rgb, 
            masked_ddata=masked_ddata, 
            resulting_mask=resulting_mask, 
            sewregions=sewregions, 
            daoregions=daoregions, 
            save_fig=save_fig,
            prefix_filename=prefix_filename
        )
        #fig = self.plot_mask(masked_ddata, resulting_mask, r_circ, sewregions, daoregions=daoregions, save_fig=save_fig)
        #fig = self.plot_mask(masked_ddata, resulting_mask, sewregions, daoregions=daoregions, save_fig=save_fig)
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

    def download_data(self):
        ctrl = self.control
        self.get_stamps()
        if ctrl.mask_stars and not ctrl.det_img:
            print_level('For mask detection image is required. Overwriting --det_img')
            ctrl.det_img = True
        if ctrl.det_img:
            self.get_detection_image()

    def create_cube(self, flam_scale=None):
        flam_scale = 1e19 if flam_scale is None else flam_scale
        ctrl = self.control

        # CUBE CHECK
        cube_filename = f'{ctrl.prefix_filename}_cube.fits'
        cube_path = join(ctrl.output_dir, cube_filename)
        if exists(cube_path) and not ctrl.redo:
            raise OSError('Cube exists!')
        
        # DOWNLOAD AND CALIBRATE DATA
        self.download_data()
        self.calibrate_stamps()
        
        # CREATE SPECTRA
        self.spectra(flam_scale=flam_scale)

        # DELETE BOGUS INFO
        cube_h = self.headers__b[0].copy()
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
        
        # MASK STARS
        if ctrl.mask_stars:
            mask_hdul = self.create_mask_hdu()     # HDUList
            mask_hdu = mask_hdul[1].copy()
            mask_hdu.header['EXTNAME'] = ('MASK', 'Boolean mask of the galaxy')
            hdu_list.append(mask_hdu)
        
        # METADATA
        meta_hdu = self.create_metadata_hdu()  # BinTableHDU
        meta_hdu.header['EXTNAME'] = 'METADATA'
        hdu_list.append(meta_hdu)

        # SAVE CUBE
        print_level(f'writting cube {cube_path}', 1, ctrl.verbose)
        fits.HDUList(hdu_list).writeto(cube_path, overwrite=True)
        print_level(f'Cube successfully created!')