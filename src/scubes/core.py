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
from astropy.coordinates import SkyCoord
from os.path import join, exists, isfile
from scipy.interpolate import RectBivariateSpline

from . import __filters_table__, __dr4_zp_cat__, __dr4_zpcorr_path__, \
    __dr5_zp_cat__, __dr5_zpcorr__

from .control import control
from .headers import get_author, get_key
from .constants import FILTER_NAMES_DR4_ZP_TABLE, CENTRAL_WAVE, METADATA_NAMES

from .utilities.io import print_level
from .utilities.splusdata import connect_splus_cloud, detection_image_hdul, get_lupton_rgb

_disable_zpcorr = True  # waiting S-PLUS iDR6...

@dataclass
class _galaxy:
    '''
    A data class representing galaxy information.

    Attributes
    ----------
    ra : :class:`~astropy.units.Quantity`
        Right Ascension of the galaxy.

    dec : :class:`~astropy.units.Quantity`
        Declination of the galaxy.

    name : str
        Name of the galaxy.

    Methods
    -------
    skycoord(frame='icrs')
        Get the `SkyCoord` object for the galaxy coordinates.

    See Also
    --------
    :class:`~astropy.coordinates.SkyCoord`
    '''    
    ra: u.quantity.Quantity
    dec: u.quantity.Quantity
    name: str

    def skycoord(self, frame='icrs'):
        '''
        Get the `SkyCoord` object for the galaxy coordinates.

        Parameters
        ----------
        frame : str, optional
            Coordinate frame (default is 'icrs').

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`
            SkyCoord object for the galaxy coordinates.
        '''      
        return SkyCoord(ra=self.ra, dec=self.dec, frame=frame, unit='deg')
    
class _control(control):
    '''
    Extended :class:`control` for handling specific arguments and directories.

    Attributes
    ----------
    args : :class:`~argparse.Namespace`
        Parsed command-line arguments.

    output_dir : str
        Output directory for storing results.

    prefix_filename : str
        Prefix for the output filenames.

    Methods
    -------
    _make_output_dir()
        Create the output directory.

    See Also
    --------
    :class:`control`
    '''
    def __init__(self, args):
        '''
        Initialize the _control object.

        Parameters
        ----------
        args : :class:`~argparse.Namespace`
            Parsed command-line arguments.
        '''
        super().__init__(args)
        self.output_dir = join(self.work_dir, self.galaxy)
        print_level(f'output_dir: {self.output_dir}', 2, self.verbose)
        print_level(f'prefix_filename: {self.prefix_filename}', 2, self.verbose)
        self._make_output_dir()

    def _make_output_dir(self):
        '''
        Create the output directory.
        '''   
        try: 
            makedirs(self.output_dir)
        except FileExistsError:
            print_level(f'{self.output_dir}: directory already exists', 2, self.verbose)    

class SCubes:
    '''
    Class for creating S-PLUS galaxy data cubes (S-CUBES).

    Attributes
    ----------
    args : :class:`~argparse.Namespace`
        Parsed command-line arguments.

    _conn : object
        Connection object to the S-PLUS Cloud.

    control : :class:`~_control`
        Control object for handling specific arguments and directories.

    galaxy : :class:`~_galaxy`
        Galaxy object representing galaxy information.

    wl__b : :class:`~numpy.ndarray`
        Array of effective wavelengths for each band.

    flam_unit : :class:`~astropy.units.Unit`
        Unit for flux density.

    fnu_unit : :class:`~astropy.units.Unit`
        Unit for flux.

    flam__b : :class:`~numpy.ndarray`
        Array of flux density values.

    fnu__b : :class:`~numpy.ndarray`
        Array of flux values.

    headers__b : list
        List of headers for each band.

    See Also
    --------
    :class:`~_control`, :class:`~_galaxy`, :class:`control`
    '''    
    def __init__(self, args):
        '''
        Initialize the SCubes object.

        Parameters
        ----------
        args : :class:`~argparse.Namespace`
            Parsed command-line arguments.
        '''        
        self._conn = None
        self.args = args
        self.control = _control(self.args)
        self.detection_image = None
        self.lupton_rgb_filename = None
        self._init_galaxy()
        self._init_spectra()

    def _init_galaxy(self):
        '''
        Initialize the _galaxy object.
        '''        
        self.galaxy = self._galaxy()
        gal = self.galaxy
        gal.ra = self.control.ra
        gal.dec = self.control.dec
        gal.name = self.control.galaxy
        self.galaxy.coords = self.galaxy.skycoord()

    def _init_spectra(self):
        '''
        Initialize the spectra arrays.
        '''        
        self.wl__b = np.array(sorted([CENTRAL_WAVE[b] for b in self.control.bands]))*u.Angstrom
        self.flam_unit = u.erg / u.s / u.cm / u.cm / u.AA
        self.fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        self.flam__b = None
        self.fnu__b = None
        self.headers__b = None

    @property
    def conn(self):
        '''
        Get the connection object to the S-PLUS Cloud.

        Returns
        -------
        object
            Connection object to the S-PLUS Cloud.
        '''        
        if self._conn is None:
            ctrl = self.control
            self._conn = connect_splus_cloud(ctrl.username, ctrl.password)
        return self._conn

    def _check_errors(self):
        '''
        Check if there are any errors in the data.

        Returns
        -------
        bool
            True if no errors, False otherwise.
        '''        
        return all([exists(_) for _ in self.wimages])

    def _get_headers_list(self, images=None, ext=1):
        '''
        Get a list of headers for the given images.

        Parameters
        ----------
        images : list, optional
            List of image filenames, by default None.
        ext : int, optional
            FITS extension number, by default 1.

        Returns
        -------
        list
            List of headers for each image.
        '''        
        images = self.images if images is None else images
        return [fits.getheader(img, ext=ext) for img in images]

    def _get_data_spectra(self, images=None, ext=1):
        '''
        Get data spectra from FITS images.

        Parameters
        ----------
        images : list of str, optional
            List of FITS image filenames, by default None.
        ext : int, optional
            FITS extension from which to extract data, by default 1.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of data spectra from the specified FITS images.
        '''        
        images = self.images if images is None else images
        return np.array([fits.getdata(img, ext=ext) for img in images])

    def _header_key_spectra(self, key):
        '''
        Extract values for a specific header key from a set of spectra headers.

        Parameters
        ----------
        key : str
            Header key to extract values.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of values corresponding to the specified header key across multiple spectra.
        '''        
        values = []
        for i, h in enumerate(self.headers__b):
            author = get_author(h)
            k = get_key(key, author)
            values.append(h[k])
        return np.array(values)

    def _m0(self):
        '''
        Get the zero-point magnitude.
        '''        
        return self._header_key_spectra('MAGZP')
    
    def _gain(self):
        '''
        Get the gain values.
        '''
        return self._header_key_spectra('GAIN')
    
    def _galaxy(self):
        '''
        Create a _galaxy object.
        '''
        c = self.control
        return _galaxy(ra=c.ra, dec=c.dec, name=c.galaxy)
    
    def check_zero_points(self):
        '''
        Check if zero-point magnitudes are available.        
        '''
        has_magzp = True
        for h in self.headers__b:
            mzp_key = get_key('MAGZP', get_author(h))
            if h.get(mzp_key, None) is None:
                has_magzp = False
                break
        return has_magzp

    def _check_write_mar_author(self, header):
        '''
            Fix MAR authorship at the header.

            Parameters
            ----------
            header : str

            Note: Gustavo said that this problem will fix for the iDR6!
        '''
        ctrl = self.control
        author = get_author(header)
        # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
        if (author is None) or (author == 'unknown'):
            print_level('Writting header key AUTHOR', 1, ctrl.verbose)
            header.set('AUTHOR', value='MAR', comment='Who ran the software')

    def _add_info_stamps_header(self):
        gal = self.galaxy
        ctrl = self.control
        # UPDATE IMAGES HEADER - WILL ADD TILE INFO AND WCS TO THE HEADERS
        for img in self.images:
            with fits.open(img, 'update') as f:
                print_level(f'{gal.name}: {ctrl.tile}: {img}: add OBJECT and TILE to header', 2, ctrl.verbose)
                w = WCS(f[1].header)
                f[1].header['OBJECT'] = gal.name
                f[1].header['TILE'] = ctrl.tile
                f[1].header.update(w.to_header())

    def get_stamps(self):
        '''
        Download stamps for each band.
        '''
        gal = self.galaxy
        ctrl = self.control
        self.stamps = []
        for filt in tqdm(list(CENTRAL_WAVE.keys()), desc=f'{gal.name} @ {ctrl.tile} - downloading', leave=True, position=0):
            fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{filt}_{ctrl.size}x{ctrl.size}_swp.fits.fz')
            self.stamps.append(fname)
            if not isfile(fname) or ctrl.force:
                kw_stamp = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, band=filt, weight=False, outfile=fname)  #, field_name=ctrl.tile)
                kw_stamp['_data_release'] = ctrl.data_release
                _ = self.conn.stamp(**kw_stamp)
            # Download_weight:
            fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{filt}_{ctrl.size}x{ctrl.size}_swpweight.fits.fz')
            self.stamps.append(fname)
            if not isfile(fname) or ctrl.force:
                kw_stamp = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, band=filt, weight=True, outfile=fname)  #, field_name=ctrl.tile)
                kw_stamp['_data_release'] = ctrl.data_release
                _ = self.conn.stamp(**kw_stamp)
        # SHORTCUTS
        self.images = [img for img in self.stamps if 'swp.' in img]
        self.wimages = [img for img in self.stamps if 'swpweight.' in img]
        self._add_info_stamps_header()
        self.headers__b = self._get_headers_list(self.images, ext=1)
        for header in self.headers__b:
            self._check_write_mar_author(header)
       
    def get_detection_image(self):
        '''
        Download the detection image.        
        '''
        gal = self.galaxy
        ctrl = self.control
        band = 'G,R,I,Z'
        self.detection_image = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{ctrl.size}x{ctrl.size}_detection.fits')
        if not isfile(self.detection_image) or ctrl.force:
            print_level(f' {gal.name} @ {ctrl.tile} - downloading detection image')
            kw = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, bands=band, option=ctrl.tile)
            kw['_data_relase'] = ctrl.data_release
            hdul = detection_image_hdul(self.conn, wcs=True, **kw)
            self._check_write_mar_author(hdul[1].header)
            hdul.writeto(self.detection_image, overwrite=ctrl.force)
                        
    def get_lupton_rgb(self):
        '''
        Download the Lupton RGB image.        
        '''
        gal = self.galaxy
        ctrl = self.control
        #conn = self.get_splusdata_conn()
        fname = join(ctrl.output_dir, f'{gal.name}_{ctrl.tile}_{ctrl.size}x{ctrl.size}.png')
        self.lupton_rgb_filename = fname
        if not isfile(fname) or ctrl.force:
            print_level(f'{gal.name} @ {ctrl.tile} - downloading RGB image')
            kw = dict(ra=gal.ra, dec=gal.dec, size=ctrl.size, option=ctrl.tile)
            kw['_data_relase'] = ctrl.data_release
            img = get_lupton_rgb(self.conn, transpose=True, **kw)
        else:
            img = Image.open(fname)
        self.lupton_rgb = img

    def get_zero_points_correction(self):
        '''
        Get corrections for zero points.
        
        XXX: 2025-03-14 - EADL@RV
            zp correction image is based on a 9200 x 9200 image,
            the valid CCD area. When creating stamps, the stamp
            FITS header maps the x0, y0 position on the original
            11000 x 11000 field image. This creates a huge problem
            on correcting stamps. 
        '''
        ctrl = self.control
        zpcorr = {}
        print_level('Getting ZP corrections for the S-PLUS bands...')
        for h in self.headers__b:
            author = get_author(h)
            band = h.get(get_key('FILTER', author))
            if '4' in ctrl.data_release:
                self.zpcorr_dir = __dr4_zpcorr_path__
                nbins = 32
            elif '5' in ctrl.data_release:
                self.zpcorr_dir = __dr5_zpcorr__[author]
                nbins = 64
            else:
                print_level(f'{ctrl.data_release}: wrong data release')
                self.remove_downloaded_data()
                sys.exit(1)
            x0, x1 = 0, 9200
            xgrid = np.linspace(x0, x1, nbins + 1)
            corrfile = join(self.zpcorr_dir, 'SPLUS_' + band + '_offsets_grid.npy')
            print_level(f'Reading ZPs corr image: {corrfile}')
            corr = np.load(corrfile)
            zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)
        self.zpcorr = zpcorr

    def get_zero_points(self):
        '''
        Get zero points from the specified table.
        '''
        ctrl = self.control
        self.zp_table = None
        if '4' in ctrl.data_release:
            self.zp_table = __dr4_zp_cat__
        elif '5' in ctrl.data_release:           
            self.zp_table = __dr5_zp_cat__
        print_level(f'Reading ZPs table: {self.zp_table}')
        zpt = pd.read_csv(self.zp_table)
        cols = [col.replace('ZP_', '') if 'ZP_' in col else col for col in zpt.columns]
        zpt.columns = cols
        # New class properties
        self.zptab = zpt[zpt['Field'].replace('_', '-') == ctrl.tile.replace('_', '-')]
        if self.zptab.size == 0:
            print_level(f'{ctrl.tile}: not found in zero-points table')
            self.remove_downloaded_data()
            sys.exit(1)
        if not _disable_zpcorr:
            self.get_zero_points_correction()

    def add_magzp_headers(self):
        '''
        Add magnitude zero-point values to the image headers.
        '''
        ctrl = self.control
        self.get_zero_points()
        print_level('Calibrating stamps...')
        headers = []
        for img, h in zip(self.images, self.headers__b):
            #h = fits.getheader(img, ext=1)
            h['TILE'] = ctrl.tile
            filtername = h['FILTER']
            if ('4' in ctrl.data_release) and (filtername in FILTER_NAMES_DR4_ZP_TABLE.keys()):
                k = FILTER_NAMES_DR4_ZP_TABLE[filtername]
            elif filtername in self.zptab.keys():
                k = filtername
            else:
                print_level(f'{ctrl.tile}: {filtername}: missing zp correction data')
                self.remove_downloaded_data()
                sys.exit(1)
            zp = float(self.zptab[k].item())
            x0 = h['X0TILE']
            y0 = h['Y0TILE']
            if not _disable_zpcorr:
                zp += round(self.zpcorr[filtername](x0, y0)[0][0], 5)
            fits.setval(img, 'MAGZP', value=zp, comment='Magnitude zero point', ext=1)
            headers.append(fits.getheader(img, ext=1))
            print_level(f'add_magzp_headers: {img}: MAGZP={zp}', level=2, verbose=ctrl.verbose)
        # updates the list of headers with the new values
        self.headers__b = headers
         
    def calibrate_stamps(self):
        '''
        Calibrate the downloaded stamps.
        '''
        if not self.check_zero_points():
            self.add_magzp_headers()

    def stamp_WCS_to_cube_header(self, header):
        '''
        Convert WCS information from stamp to cube header.

        Parameters
        ----------
        header : :class:`~astropy.io.fits.Header`
            FITS header containing WCS information.

        Returns
        -------
        :class:`~astropy.io.fits.Header`
            Cube header with updated WCS information.
        '''        
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
        '''
        Create a metadata table HDU.

        Parameters
        ----------
        keys : list, optional
            List of header keys, by default None.

        Returns
        -------
        :class:`~astropy.io.fits.hdu.table.BinTableHDU`
            Metadata table HDU.
        '''  
        tab = []
        names = []
        items = ['filter', 'central_wave', 'pivot_wave']        
        for k in items:
            if k in __filters_table__.colnames:
                names.append(METADATA_NAMES[k])
                v = __filters_table__[k]
                tab.append(v)

        # PSFFWHM
        list_values = []
        for h in self.headers__b:
            k = get_key('PSFFWHM', get_author(h))
            val = h.get(k, None)
            if val is None:
                raise ValueError(f'Missing value: PSFFWHM: {val}')
            else:
                list_values.append(val)
        tab.append(list_values)
        names.append('PSFFWHM')

        meta_tab = Table(tab, names=names)
        meta_hdu = fits.BinTableHDU(meta_tab)
        return meta_hdu

    def spectra(self, flam_scale=None):
        '''
        Calculate the spectra arrays.

        Parameters
        ----------
        flam_scale : float, optional
            Scaling factor for flux density, by default None.
        '''        
        flam_scale = 1e-19 if flam_scale is None else flam_scale
        _c = const.c
        scale = (1/flam_scale)
    
        '''
        XXX: 2025-03-14 - EADL
            The zp magnitude used now is the value of the central 
            pixel of the galaxy. Since we have a zp correction for
            each pixel, we can create a m0 map, resulting in a f0 
            map.
        '''
        #Jy to to erg/s/cm/cm/Hz
        Jy2fnu = - 2.5*(np.log10(3631) - 23)  # 48.5999343777177...
        
        # MAGZP
        self.m0__b = self._m0()  # mAB
        self.f0__b = np.power(10, -0.4*(Jy2fnu + self.m0__b))
        
        # from e- counts to erg/s/cm/cm/A
        self.data__byx = self._get_data_spectra(self.images, 1)
        self.fnu__byx = self.data__byx*self.f0__b[:, None, None]*self.fnu_unit
        self.flam__byx = scale*(self.fnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

        if self._check_errors():
            self.gain__b = self._gain()
            gain__byx = self.gain__b[:, None, None]
            dataclip__byx = np.abs(self.data__byx)
            weidata__byx = np.abs(self._get_data_spectra(self.wimages, 1))
            dataerr__byx = np.sqrt(1/weidata__byx + dataclip__byx/gain__byx)
            self.efnu__byx = dataerr__byx*self.f0__b[:, None, None]*self.fnu_unit
            self.eflam__byx = scale*(self.efnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

    def download_data(self):
        ctrl = self.control
        self.get_stamps()
        #if ctrl.mask_stars and not ctrl.det_img:
        #    print_level('For mask detection image is required. Overwriting --det_img')
        #    ctrl.det_img = True
        if ctrl.det_img:
            self.get_detection_image()

    def create_weights_mask_hdu(self):
        '''
        Create a weights mask HDU.

        Returns
        -------
        :class:`~astropy.io.fits.hdu.image.ImageHDU`
            Weight mask HDU.
        '''
        w__byx = self._get_data_spectra(self.wimages, 1)
        wmask__byx = np.where(w__byx < 0, 1, 0)
        wmask__yx = wmask__byx.sum(axis=0)
        wmask_hdu = fits.ImageHDU(wmask__yx)
        wmask_hdu.header['EXTNAME'] = ('WEIMASK', 'Sum of negative weight pixels (from 1 to 12)')
        return wmask_hdu
    
    def remove_downloaded_data(self):
        '''
        Remove downloaded stamp, detection image, and Lupton RGB image files.
        '''        
        print_level('Removing downloaded data')
        ctrl = self.control
        for f, wf in zip(self.images, self.wimages):
            print_level(f'removing file {f}', 1, ctrl.verbose)
            remove(f)
            print_level(f'removing file {f}', 1, ctrl.verbose)
            remove(wf)
        files_to_remove = [self.detection_image, self.lupton_rgb_filename]
        for f in files_to_remove:
            if f is not None:
                if isfile(f):
                    print_level(f'removing file {f}', 1, ctrl.verbose)
                    remove(f)
                else:
                    print_level(f'file {f} do not exists', 1, ctrl.verbose)

    def create_cube(self, flam_scale=None):
        '''
        Create a data cube from S-PLUS galaxy stamps.

        Parameters
        ----------
        flam_scale : float, optional
            Scaling factor for flux density, by default None.

        Raises
        ------
        OSError
            Raises an error if the cube already exists and redo is not specified.
        '''        
        flam_scale = 1e-19 if flam_scale is None else flam_scale
        ctrl = self.control

        # CUBE CHECK
        #cube_filename = f'{ctrl.prefix_filename}_cube.fits'
        cube_filename = f'{self.galaxy.name}_cube.fits'
        cube_path = join(ctrl.output_dir, cube_filename)
        self.cube_path = cube_path
        if exists(cube_path) and not ctrl.redo:
            raise OSError('Cube exists!')
        
        # DOWNLOAD AND CALIBRATE DATA
        self.download_data()
        self.calibrate_stamps()
        
        # CREATE SPECTRA
        self.spectra(flam_scale=flam_scale)

        # DELETE BOGUS INFO
        cube_h = self.headers__b[0].copy()
        for _k in ['FILTER', 'MAGZP', 'NCOMBINE', 'GAIN', 'PSFFWHM']:
            k = get_key(_k, get_author(cube_h))
            if cube_h.get(k, None) is not None:
                print_level(f'create_cube: deleting header key {k}', 2, ctrl.verbose)
                del cube_h[k]    
        
        # UPDATE WCS IN HEADER
        cube_h.update(self.stamp_WCS_to_cube_header(cube_h))
        
        # CREATE CUBE
        prim_hdu = fits.PrimaryHDU()
        prim_hdu.header['TILE'] = ctrl.tile
        prim_hdu.header['GALAXY'] = self.galaxy.name
        prim_hdu.header['SIZE'] = (ctrl.size, 'Side of the stamp in pixels')
        for _k in ['X0TILE', 'Y0TILE']:
            prim_hdu.header[_k] = cube_h[_k]
        prim_hdu.header['RA'] = ctrl.ra
        prim_hdu.header['DEC'] = ctrl.dec
        flam_hdu = fits.ImageHDU(self.flam__byx, cube_h)
        flam_hdu.header['EXTNAME'] = ('DATA', 'Name of the extension')
        hdu_list = [prim_hdu, flam_hdu]
        if self._check_errors():
            eflam_hdu = fits.ImageHDU(self.eflam__byx, cube_h)
            eflam_hdu.header['EXTNAME'] = ('ERRORS', 'Name of the extension')
            hdu_list.append(eflam_hdu)
        for hdu in hdu_list[1:]:
            hdu.header['BSCALE'] = (flam_scale, 'Linear factor in scaling equation')
            hdu.header['BZERO'] = (0, 'Zero point in scaling equation') 
            hdu.header['BUNIT'] = (f'{self.flam_unit}', 'Physical units of the array values')       

        # MASK WEIGHTS
        hdu_list.append(self.create_weights_mask_hdu())
            
        # MASK STARS
        '''
        if ctrl.mask_stars:
            from .mask_stars import maskStars
            
            self.get_lupton_rgb()
            mask = maskStars(args=self.args, detection_image=self.detection_image, lupton_rgb=self.lupton_rgb, output_dir=ctrl.output_dir)
            # HDUList
            mask_hdul = mask.hdul  
            mask_hdu = mask_hdul[1].copy()

            mask_hdu.header['EXTNAME'] = ('STARMASK', 'Boolean mask of stars along the FOV')
            hdu_list.append(mask_hdu)
        '''
        
        # METADATA
        meta_hdu = self.create_metadata_hdu()  # BinTableHDU
        meta_hdu.header['EXTNAME'] = 'METADATA'
        hdu_list.append(meta_hdu)

        # SAVE CUBE
        print_level(f'writting cube {cube_path}', 1, ctrl.verbose)
        fits.HDUList(hdu_list).writeto(cube_path, overwrite=True)
        print_level(f'Cube successfully created!')

        self.remove_downloaded_data() if ctrl.remove_downloaded_data else None