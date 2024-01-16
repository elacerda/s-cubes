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

from .control import control
from .mask_stars import maskStars
from .headers import get_author, get_key
from .constants import WAVE_EFF, NAMES_CORRESPONDENT 

from .utilities.io import print_level
from .utilities.splusdata import connect_splus_cloud, detection_image_hdul, get_lupton_rgb

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

    redshift : float
        Redshift of the galaxy.

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
    redshift: float

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

    Methods
    -------
    _init_galaxy()
        Initialize the _galaxy object.

    _init_spectra()
        Initialize the spectra arrays.

    _check_errors()
        Check if there are any errors in the data.

    _get_headers_list(images=None, ext=1)
        Get a list of headers for the given images.

    _get_data_spectra(images=None, ext=1)
        Get the data arrays for the given images.

    _header_key_spectra(key)
        Get header values for a specific key.

    _m0()
        Get the zero-point magnitude.

    _gain()
        Get the gain values.

    _effexptime()
        Get the effective exposure time.

    _galaxy()
        Create a _galaxy object.

    check_zero_points()
        Check if zero-point magnitudes are available.

    get_stamps()
        Download stamps for each band.

    get_detection_image()
        Download the detection image.

    get_lupton_rgb()
        Download the Lupton RGB image.

    get_zero_points_correction()
        Get corrections for zero points.

    get_zero_points()
        Get zero points from the specified table.

    add_magzp_headers()
        Add magnitude zero-point values to the image headers.

    calibrate_stamps()
        Calibrate the downloaded stamps.

    stamp_WCS_to_cube_header(header)
        Convert stamp WCS to cube header.

    create_metadata_hdu(keys=None)
        Create a metadata table HDU.

    spectra(flam_scale=None)
        Calculate the spectra arrays.

    download_data()
        Download stamps, detection image, and Lupton RGB image.

    create_weights_mask_hdu()
        Create a weights mask HDU.

    remove_downloaded_data()
        Remove downloaded stamp, detection image, and Lupton RGB image files.

    create_cube(flam_scale=None)
        Create the data cube.

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
        gal.redshift = self.control.specz
        self.galaxy.coords = self.galaxy.skycoord()

    def _init_spectra(self):
        '''
        Initialize the spectra arrays.
        '''        
        self.wl__b = np.array([WAVE_EFF[b] for b in self.control.bands])*u.Angstrom
        self.flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
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
    
    def _effexptime(self):
        '''
        Get the effective exposure time.
        '''
        return self._header_key_spectra('EFFTIME')
    
    def _galaxy(self):
        '''
        Create a _galaxy object.
        '''
        c = self.control
        return _galaxy(ra=c.ra, dec=c.dec, name=c.galaxy, redshift=c.specz)
    
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

    def get_stamps(self):
        '''
        Download stamps for each band.
        '''
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
            hdul = detection_image_hdul(self.conn, wcs=True, **kw)
            author = get_author(hdul[1].header)

            # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
            if author == 'unknown':
                print_level('Writting header key AUTHOR to detection image', 1, ctrl.verbose)
                author = get_author(self.headers__b[0])
                hdul[1].header.set('AUTHOR', value=author, comment='Who ran the software')

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
            img = get_lupton_rgb(self.conn, transpose=True, **kw)
        else:
            img = Image.open(fname)
        self.lupton_rgb = img

    def get_zero_points_correction(self):
        '''
        Get corrections for zero points.
        '''
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
        '''
        Get zero points from the specified table.
        '''
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
        '''
        Add magnitude zero-point values to the image headers.
        '''
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
        '''
        Calibrate the downloaded stamps.
        '''
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
    
        self.m0__b = self._m0()
        self.gain__b = self._gain()
        self.effexptime__b = self._effexptime()
        self.data__byx = self._get_data_spectra(self.images, 1)   
        self.f0__b = np.power(10, -0.4*(48.6 + self.m0__b))
        self.fnu__byx = self.data__byx*self.f0__b[:, None, None]*self.fnu_unit
        self.flam__byx = scale*(self.fnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

        if self._check_errors():
            weidata__byx = np.abs(self._get_data_spectra(self.wimages, 1))
            dataclip__byx = np.abs(self.data__byx)
            gain__byx = self.gain__b[:, None, None]
            dataerr__byx = np.sqrt(1/weidata__byx + dataclip__byx/gain__byx)
            self.efnu__byx = dataerr__byx*self.f0__b[:, None, None]*self.fnu_unit
            self.eflam__byx = scale*(self.efnu__byx*_c/self.wl__b[:, None, None]**2).to(self.flam_unit).value

    def download_data(self):
        ctrl = self.control
        self.get_stamps()
        if ctrl.mask_stars and not ctrl.det_img:
            print_level('For mask detection image is required. Overwriting --det_img')
            ctrl.det_img = True
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
        print_level('Removing downloaded data"')
        ctrl = self.control
        for f, wf in zip(self.images, self.wimages):
            print_level(f'removing file {f}', 1, ctrl.verbose)
            remove(f)
            print_level(f'removing file {f}', 1, ctrl.verbose)
            remove(wf)
        files_to_remove = [self.detection_image, self.lupton_rgb_filename]
        for f in files_to_remove:
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
        prim_hdu.header['SPECZ'] = (ctrl.specz, 'Spectroscopic redshift')
        prim_hdu.header['PHOTZ'] = ('', 'Photometric redshift')
        prim_hdu.header['TILE'] = ctrl.tile
        prim_hdu.header['GALAXY'] = self.galaxy.name
        for _k in ['X0TILE', 'Y0TILE']:
            prim_hdu.header[_k] = cube_h[_k]
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
        if ctrl.mask_stars:
            self.get_lupton_rgb()
            mask = maskStars(args=self.args, detection_image=self.detection_image, lupton_rgb=self.lupton_rgb, output_dir=ctrl.output_dir)
            # HDUList
            mask_hdul = mask.hdul  
            mask_hdu = mask_hdul[1].copy()

            mask_hdu.header['EXTNAME'] = ('STARMASK', 'Boolean mask of stars along the FOV')
            hdu_list.append(mask_hdu)
        
        # METADATA
        meta_hdu = self.create_metadata_hdu()  # BinTableHDU
        meta_hdu.header['EXTNAME'] = 'METADATA'
        hdu_list.append(meta_hdu)

        # SAVE CUBE
        print_level(f'writting cube {cube_path}', 1, ctrl.verbose)
        fits.HDUList(hdu_list).writeto(cube_path, overwrite=True)
        print_level(f'Cube successfully created!')

        self.remove_downloaded_data() if ctrl.remove_downloaded_data else None