import sys
import numpy as np
from os.path import isfile
from astropy.io import fits
from argparse import Namespace
from copy import deepcopy as copy
from astropy.visualization import make_lupton_rgb
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from .io import print_level
from .sky import get_iso_sky
from ..constants import METADATA_NAMES, BANDS

class tupperware_none(Namespace):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        r = self.__dict__.get(attr, None)
        return r

# create filters indexes list
def _parse_filters_tuple(f_tup, filters):
    if isinstance(f_tup, list):
        f_tup = tuple(f_tup)
    i_f = []
    if isinstance(f_tup, tuple):
        for f in f_tup:
            if isinstance(f, str):
                i_f.append(filters.index(f))
            else:
                i_f.append(f)
    else:
        if isinstance(f_tup, str):
            i_f.append(filters.index(f_tup))
        else:
            i_f.append(f_tup)
    return i_f

def _parse_filters_tuple(f_tup, filters):
    if isinstance(f_tup, list):
        f_tup = tuple(f_tup)
    i_f = []
    if isinstance(f_tup, tuple):
        for f in f_tup:
            i_f.append(f)
    else:
        i_f.append(f_tup)
    return i_f

def get_distance(x, y, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray`)
    of the distance from the center ``(x0, y0)`` in pixels,
    assuming a projected disk.

    Parameters
    ----------
    x : array
        X coordinates to get the pixel distances.

    y : array
        y coordinates to get the pixel distances.

    x0 : float
        X coordinate of the origin.

    y0 : float
        Y coordinate of the origin.

    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis.

    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).

    Returns
    -------
    pixel_distance : array
        Pixel distances.
    '''
    y = np.asarray(y) - y0
    x = np.asarray(x) - x0
    x2 = x**2
    y2 = y**2
    xy = x * y

    a_b = 1.0/ba
    cos_th = np.cos(pa)
    sin_th = np.sin(pa)

    A1 = cos_th ** 2 + a_b ** 2 * sin_th ** 2
    A2 = -2.0 * cos_th * sin_th * (a_b ** 2 - 1.0)
    A3 = sin_th ** 2 + a_b ** 2 * cos_th ** 2

    return np.sqrt(A1 * x2 + A2 * xy + A3 * y2)

def make_RGB_tom(flux__lyx,
        rgb=(7, 5, 9), rgb_f=(1, 1, 1), 
        pminmax=(5, 95), im_max=255, 
        # astropy.visualization.make_lupton_rgb() input vars
        minimum=(0, 0, 0), Q=0, stretch=10):
    # get filters index(es)
    pmin, pmax = pminmax
    RGB = []
    for f_tup in rgb:
        # get fluxes list
        if isinstance(f_tup, list):
            f_tup = tuple(f_tup)
        i_f = []
        if isinstance(f_tup, tuple):
            for f in f_tup:
                i_f.append(f)
        else:
            i_f.append(f_tup)
        C = copy(flux__lyx[i_f, :, :]).sum(axis=0)
        # percentiles
        Cmin, Cmax = np.nanpercentile(C, pmin), np.nanpercentile(C, pmax)
        # calc color intensities
        RGB.append(im_max*(C - Cmin)/(Cmax - Cmin))
    R, G, B = RGB
    # filters factors
    fR, fG, fB = rgb_f
    # make RGB image
    return make_lupton_rgb(fR*R, fG*G, fB*B, Q=Q, minimum=minimum, stretch=stretch)

def get_image_distance(shape, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray`)
    of the distance from the center ``(x0, y0)`` in pixels,
    assuming a projected disk.

    Parameters
    ----------
    shape : (float, float)
        Shape of the image to get the pixel distances.

    x0 : float
        X coordinate of the origin.

    y0 : float
        Y coordinate of the origin.

    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis. Defaults to ``0.0``.

    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`). Defaults to ``1.0``.

    Returns
    -------
    pixel_distance : 2-D array
        Image containing the distances.

    See also
    --------
    :func:`get_distance`

    '''
    y, x = np.indices(shape)
    return get_distance(x, y, x0, y0, pa, ba)

def radial_profile(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0, mask=None, mode='mean', return_npts=False):
    '''
    Calculate the radial profile of an N-D image.

    Parameters
    ----------
    prop : array
        Image of property to calculate the radial profile.

    bin_r : array
        Semimajor axis bin boundaries in units of ``rad_scale``.

    x0 : float
        X coordinate of the origin.

    y0 : float
        Y coordinate of the origin.

    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis.

    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).

    rad_scale : float, optional
        Scale of the bins, in pixels. Defaults to 1.0.

    mask : array, optional
        Mask containing the pixels to use in the radial profile.
        Must be bidimensional and have the same shape as the last
        two dimensions of ``prop``. Default: no mask.

    mode : string, optional
        One of:
            * ``'mean'``: Compute the mean inside the radial bins (default).
            * ``'median'``: Compute the median inside the radial bins.
            * ``'sum'``: Compute the sum inside the radial bins.
            * ``'var'``: Compute the variance inside the radial bins.
            * ``'std'``: Compute the standard deviation inside the radial bins.

    return_npts : bool, optional
        If set to ``True``, also return the number of points inside
        each bin. Defaults to ``False``.


    Returns
    -------
    radProf : [masked] array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
        If ``prop`` is a masked aray, this and ``npts`` will be
        a masked array as well.

    npts : [masked] array, optional
        The number of points inside each bin, only if ``return_npts``
        is set to ``True``.


    See also
    --------
    :func:`get_image_distance`
    '''
    def red(func, x, fill_value):
        if x.size == 0: return fill_value, fill_value
        if x.ndim == 1: return func(x), len(x)
        return func(x, axis=-1), x.shape[-1]

    imshape = prop.shape[-2:]
    nbins = len(bin_r) - 1
    new_shape = prop.shape[:-2] + (nbins,)
    r__yx = get_image_distance(imshape, x0, y0, pa, ba) / rad_scale
    if mask is None:
        mask = np.ones(imshape, dtype=bool)
    if mode == 'mean':
        reduce_func = np.mean
    elif mode == 'median':
        reduce_func = np.median
    elif mode == 'sum':
        reduce_func = np.sum
    elif mode == 'var':
        reduce_func = np.var
    elif mode == 'std':
        reduce_func = np.std
    else:
        raise ValueError('Invalid mode: %s' % mode)

    if isinstance(prop, np.ma.MaskedArray):
        n_bad = prop.mask.astype('int')
        max_bad = 1.0
        while n_bad.ndim > 2:
            max_bad *= n_bad.shape[0]
            n_bad = n_bad.sum(axis=0)
        mask = mask & (n_bad / max_bad < 0.5)
        prop_profile = np.ma.masked_all(new_shape)
        npts = np.ma.masked_all((nbins,))
        prop_profile.fill_value = prop.fill_value
        reduce_fill_value = np.ma.masked
    else:
        prop_profile = np.empty(new_shape)
        npts = np.empty((nbins,))
        reduce_fill_value = np.nan
    if mask.any():
        dist_flat = r__yx[mask]
        dist_idx = np.digitize(dist_flat, bin_r)
        prop_flat = prop[...,mask]
        for i in range(0, nbins):
            prop_profile[..., i], npts[i] = red(reduce_func, prop_flat[..., dist_idx == i+1], reduce_fill_value)

    if return_npts:
        return prop_profile, npts
    return prop_profile

class read_scube:
    '''
    Class for reading and processing data from an astronomical data cube (SCUBE).

    Parameters
    ----------
    filename : str
        The path to the FITS file to be read and processed.

    Attributes
    ----------
    filename : str
        The path to the FITS file.
    
    _hdulist : astropy.io.fits.HDUList
        The list of HDU (Header Data Units) from the FITS file.
    
    wcs : astropy.wcs.WCS
        The World Coordinate System (WCS) for the data.
    
    cent_coord : astropy.coordinates.SkyCoord
        The central sky coordinates of the object.
    
    x0, y0 : float
        Pixel coordinates of the central sky position.
    
    i_x0, i_y0 : int
        Integer pixel coordinates of the central sky position.
    
    mag_arcsec2__lyx : np.ndarray
        The magnitude per square arcsecond for each layer in the data cube.
    
    emag_arcsec2__lyx : np.ndarray
        The error in the magnitude per square arcsecond.
    
    pa, ba : float
        Position angle (pa) and axis ratio (ba) for pixel distance calculations.
    
    pixel_distance__yx : np.ndarray
        Pixel distance array from the central coordinate, adjusted by pa and ba.
    
    mask_stars_filename : str
        Filename for the star mask generated in `source_extractor()`.
    
    detection_image_filename : str
        Filename for the detection image used in `source_extractor()`.
    
    mask_stars_hdul : astropy.io.fits.HDUList
        The HDUList containing the star mask data.
    
    detection_image_hdul : astropy.io.fits.HDUList
        The HDUList containing the detection image data.
    
    Methods
    -------
    _read()
        Reads the FITS file and loads the data.
    
    _init_wcs()
        Initializes the WCS (World Coordinate System) for the data.
    
    _init_centre()
        Calculates the central coordinates of the object in pixel space.
    
    _mag_values()
        Computes magnitude values and their errors for each layer in the data cube.
    
    _init()
        Initializes WCS, central coordinates, and magnitude values.
    
    lRGB_image(rgb, rgb_f, pminmax, im_max, minimum, Q, stretch)
        Creates an RGB image from the data cube using specified filters.
    
    source_extractor(sextractor, username, password, class_star, satur_level, back_size, detect_thresh, estimate_fwhm, force, verbose)
        Runs source extraction using SExtractor on the data cube.
    
    get_iso_sky(isophotal_limit, isophotal_medsize, stars_mask, n_sigma, n_iter, clip_neg)
        Estimates the sky flux from the data cube using isophotal limits.
    
    mask_optimal()
        Generates a mask for optimal data handling based on flux and error values.
    '''
    def __init__(self, filename):
        '''
        Initialize the read_scube class by reading the FITS file and initializing attributes.

        Parameters
        ----------
        filename : str
            The path to the FITS file to be read.
        '''        
        self.filename = filename
        self._read()
        self._init()
    
    def _read(self): 
        '''
        Reads the FITS file and loads the data into the HDUList.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        '''              
        try:
            self._hdulist = fits.open(self.filename)
        except FileNotFoundError:
            print_level(f'{self.filename} - file not found')
            sys.exit()

    def _init_wcs(self):
        '''
        Initializes the World Coordinate System (WCS) from the data header.
        '''        
        self.wcs = WCS(self.data_header, naxis=2)

    def _init_centre(self):
        '''
        Calculates the central coordinates of the object in pixel space using WCS.
        '''        
        self.cent_coord = SkyCoord(self.ra, self.dec, unit=('deg', 'deg'), frame='icrs')
        self.x0, self.y0 = self.wcs.world_to_pixel(self.cent_coord)
        self.i_x0 = int(self.x0)
        self.i_y0 = int(self.y0)

    def _mag_values(self):
        '''
        Computes the magnitude per square arcsecond and corresponding errors from the flux values.
        '''        
        a = 1/(2.997925e18*3631.0e-23*self.pixscale**2)
        x = a*(self.flux__lyx*self.pivot_wave[:, np.newaxis, np.newaxis]**2)
        self.mag_arcsec2__lyx = -2.5*np.log10(x)
        self.emag_arcsec2__lyx = (2.5*np.log10(np.exp(1)))*self.eflux__lyx/self.flux__lyx

    def _init(self):
        '''
        Initializes the class by setting WCS, central coordinates, and magnitude values.
        Also calculates the pixel distance from the central coordinates.
        '''        
        self._init_wcs()
        self._init_centre()
        self._mag_values()
        self.pa, self.ba = 0, 1
        self.pixel_distance__yx = get_image_distance(self.weimask__yx.shape, x0=self.x0, y0=self.y0, pa=self.pa, ba=self.ba)

    def lRGB_image(
        self, rgb=('rSDSS', 'gSDSS', 'iSDSS'), rgb_f=(1, 1, 1), 
        pminmax=(5, 95), im_max=255, 
        # astropy.visualization.make_lupton_rgb() input vars
        minimum=(0, 0, 0), Q=0, stretch=10):
        '''
        Creates an RGB image from the data cube using specified filters.

        Parameters
        ----------
        rgb : tuple of str or tuple of int, optional
            Tuple specifying the filters to use for the red, green, and blue channels (default is ('rSDSS', 'gSDSS', 'iSDSS')).
        
        rgb_f : tuple of float, optional
            Scaling factors for the red, green, and blue channels (default is (1, 1, 1)).
        
        pminmax : tuple of int, optional
            Percentiles for scaling the RGB intensities (default is (5, 95)).
        
        im_max : int, optional
            Maximum intensity value for the RGB image (default is 255).
        
        minimum : tuple of float, optional
            Minimum values for scaling the RGB channels (default is (0, 0, 0)).
        
        Q : float, optional
            Parameter for controlling the contrast in the Lupton RGB scaling (default is 0).
        
        stretch : float, optional
            Stretch factor for enhancing the RGB intensities (default is 10).

        Returns
        -------
        np.ndarray
            3D array representing the RGB image with shape (height, width, 3).
        '''
        # check filters
        if len(rgb) != 3:
            return None

        # check factors
        if isinstance(rgb_f, tuple) or isinstance(rgb_f, list):
            N = len(rgb_f)
            if N != 3:
                if N == 1:
                    f = rgb_f[0]
                    rgb_f = (f, f, f)
                else:
                    # FAIL
                    return None
        else:
            rgb_f = (rgb_f, rgb_f, rgb_f)

        i_rgb = [_parse_filters_tuple(x, self.filters) for x in rgb]

        return make_RGB_tom(
            self.flux__lyx, rgb=i_rgb, rgb_f=rgb_f, 
            pminmax=pminmax, im_max=im_max, 
            minimum=minimum, Q=Q, stretch=stretch
        )
    
    def source_extractor(self, 
                         sextractor, username, password, 
                         class_star=0.25, satur_level=1600, back_size=64, 
                         detect_thresh=1.1, estimate_fwhm=False,
                         force=False, verbose=0):
        '''
        Runs source extraction on the data cube using SExtractor.

        Parameters
        ----------
        sextractor : str
            Path to the SExtractor executable.
        
        username : str
            Username for accessing cloud services or external databases.
        
        password : str
            Password for accessing cloud services or external databases.
        
        class_star : float, optional
            Threshold for classifying objects as stars (default is 0.25).
        
        satur_level : float, optional
            Saturation level for the image (default is 1600).
        
        back_size : int, optional
            Background size parameter for SExtractor (default is 64).
        
        detect_thresh : float, optional
            Detection threshold for SExtractor (default is 1.1).
        
        estimate_fwhm : bool, optional
            If True, estimate the full width at half maximum (FWHM) of sources (default is False).
        
        force : bool, optional
            If True, force re-running source extraction (default is False).
        
        verbose : int, optional
            Verbosity level of the source extraction process (default is 0).
        '''        
        from ..headers import get_author
        from ..mask_stars import maskStars
        from .splusdata import connect_splus_cloud, detection_image_hdul
        from .utils import _get_lupton_RGB, sex_mask_stars_cube_argsparse

        args = tupperware_none()
        args.sextractor = sextractor
        args.verbose = verbose
        args.class_star = class_star
        args.back_size = back_size
        args.detect_thresh = detect_thresh
        args.estimate_fwhm = estimate_fwhm
        args.satur_level = satur_level
        args.force = force
        args.no_interact = True
        args.username = username
        args.password = password
        args.cube_path = self.filename
        args = sex_mask_stars_cube_argsparse(args)

        conn = connect_splus_cloud(username, password)
        detection_image = f'{self.galaxy}_detection.fits'

        if not isfile(detection_image) or force:
            print_level(f'{self.galaxy} @ {self.tile} - downloading detection image')
            kw = dict(ra=self.ra, dec=self.dec, size=self.size, bands=BANDS, option=self.tile)
            hdul = detection_image_hdul(conn, **kw)

            author = get_author(hdul[1].header)

            # ADD AUTHOR TO HEADER IF AUTHOR IS UNKNOWN
            if author == 'unknown':
                author = 'scubes'
                hdul[1].header.set('AUTHOR', value=author, comment='Who ran the software')
            # SAVE DETECTION FITS
            hdul.writeto(detection_image, overwrite=force)
        else:
            print_level('Detection file exists.')

        _ = maskStars(args=args, detection_image=detection_image, lupton_rgb=_get_lupton_RGB(conn, args, save_img=False), output_dir='.')
        self.mask_stars_filename = _.filename
        self.detection_image_filename = _.detection_image
        self.mask_stars_hdul = _.hdul
        self.detection_image_hdul = _.detection_image_hdul

    def get_iso_sky(self, ref_mag_filt='rSDSS', isophotal_limit=25, isophotal_medsize=10, stars_mask=None, n_sigma=3, n_iter=5, clip_neg=False):
        '''
        Estimates the sky flux using isophotal limits and clipping outliers in the flux data.

        Parameters
        ----------
        ref_mag_filt : str or int, optional
            Filter name or index number for the reference magnitude for the 
            isophotal limits evaluation.

        isophotal_limit : float, optional
            Threshold value for selecting sky pixels (default is 25).
        
        isophotal_medsize : int, optional
            Size of the median filter for smoothing the mask (default is 10).
        
        stars_mask : np.ndarray, optional
            Mask indicating the positions of stars (default is None).
        
        n_sigma : float, optional
            Sigma clipping threshold (default is 3).
        
        n_iter : int, optional
            Number of iterations for sigma clipping (default is 5).
        
        clip_neg : bool, optional
            If True, clip negative and positive outliers (default is False).

        Returns
        -------
        dict
            Dictionary containing the sky flux information and masks.
        '''        
        try:
            i_l = self.get_filter_i(ref_mag_filt)
        except ValueError:
            i_l = ref_mag_filt
        except:
            print_level(f'ref_mag_filt: {ref_mag_filt}: inexistent', )
            return {}

        try:
            reference_mag_img__yx = self.mag_arcsec2__lyx[i_l]
        except:
            print_level(f'ref_mag_filt: {ref_mag_filt}: inexistent', )
            return {}
        
        flux__lyx = self.flux__lyx
        reference_mag_img__yx = np.where(flux__lyx[i_l] <= 0, 99, reference_mag_img__yx)
        return get_iso_sky(
            refmag__yx=reference_mag_img__yx, flux__lyx=flux__lyx, 
            isophotal_limit=isophotal_limit, isophotal_medsize=isophotal_medsize, 
            stars_mask=stars_mask, n_sigma=n_sigma, n_iter=n_iter, clip_neg=clip_neg,
        )

    def mask_optimal(self):
        '''
        Creates an optimal mask for the data cube based on flux, error, and weight information.

        The mask is generated by checking the following conditions:
        - The weight image (`weimask__lyx`) is greater than zero.
        - The flux values are non-negative.
        - The error values are finite.

        Returns
        -------
        np.ndarray
            A boolean array where `True` indicates valid pixels (not masked) and `False` indicates masked pixels.
        '''        
        f = self._hdulist['DATA'].data
        ef = self._hdulist['ERRORS'].data
        wei = self.weimask__lyx
        return np.bitwise_or(wei>0, f<=0, ~(np.isfinite(ef)))
    
    def get_filter_i(self, filt):
        return self.filters.index(filt)
    
    @property
    def weimask__lyx(self):
        return np.broadcast_to(self.weimask__yx, (len(self.filters), self.size, self.size))

    @property
    def primary_header(self):
        return self._hdulist['PRIMARY'].header

    @property
    def data_header(self):
        return self._hdulist['DATA'].header

    @property
    def metadata(self):
        return self._hdulist['METADATA'].data
    
    @property
    def filters(self):
        return self.metadata[METADATA_NAMES['filter']].tolist()

    @property
    def central_wave(self):
        return self.metadata[METADATA_NAMES['central_wave']]

    @property
    def pivot_wave(self):
        return self.metadata[METADATA_NAMES['pivot_wave']]
    
    @property
    def psf_fwhm(self):
        return self.metadata[METADATA_NAMES['psf_fwhm']]

    @property
    def tile(self):
        return self.primary_header.get('TILE', None)
    
    @property
    def galaxy(self):
        return self.primary_header.get('GALAXY', None)
    
    @property
    def size(self):
        return self.primary_header.get('SIZE', None)
       
    @property
    def ra(self):
        return self.primary_header.get('RA', None)
    
    @property
    def dec(self):
        return self.primary_header.get('DEC', None)

    @property
    def x0tile(self):
        return self.primary_header.get('X0TILE', None)

    @property
    def y0tile(self):
        return self._hdulist[0].header['Y0TILE']

    @property
    def pixscale(self):
        return self.data_header.get('PIXSCALE', 0.55)

    @property 
    def weimask__yx(self):
        return self._hdulist['WEIMASK'].data

    @property 
    def flux__lyx(self):
        return self._hdulist['DATA'].data

    @property 
    def eflux__lyx(self):
        return self._hdulist['ERRORS'].data
    
    @property
    def n_x(self):
        return self.data_header.get('NAXIS1', None)
    
    @property
    def n_y(self):
        return self.data_header.get('NAXIS2', None)
    
    @property
    def n_filters(self):
        return self.data_header['NAXIS3']
    
    @property
    def SN__lyx(self):
        return self.flux__lyx/self.eflux__lyx

    @property
    def mag__lyx(self):
        return self.mag_arcsec2__lyx

    @property
    def emag__lyx(self):
        return self.emag_arcsec2__lyx