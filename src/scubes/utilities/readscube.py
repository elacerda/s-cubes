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

class read_scube:
    def __init__(self, filename):
        self.filename = filename
        self._read()
        self._init()
    
    def _read(self):       
        try:
            self._hdulist = fits.open(self.filename)
        except FileNotFoundError:
            print_level(f'{self.filename} - file not found')
            sys.exit()

    def _init_wcs(self):
        self.wcs = WCS(self.data_header, naxis=2)

    def _init_centre(self):
        self.cent_coord = SkyCoord(self.ra, self.dec, unit=('deg', 'deg'), frame='icrs')
        self.x0, self.y0 = self.wcs.world_to_pixel(self.cent_coord)
        self.i_x0 = int(self.x0)
        self.i_y0 = int(self.y0)

    def _mag_values(self):
        a = 1/(2.997925e18*3631.0e-23*self.pixscale**2)
        x = a*(self.flux__lyx*self.pivot_wave[:, np.newaxis, np.newaxis]**2)
        self.mag_arcsec2__lyx = -2.5*np.log10(x)
        self.emag_arcsec2__lyx = (2.5*np.log10(np.exp(1)))*self.eflux__lyx/self.flux__lyx

    def _init(self):
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
        make RGB
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

        #################
        ### RGB image ###
        #################
        # get filters index(es)
        pmin, pmax = pminmax
        RGB = []
        for _c in rgb:
            # get filters fluxes
            i_c = _parse_filters_tuple(_c, self.filters)
            C = copy(self.flux__lyx[i_c, :, :]).sum(axis=0)
            # percentiles
            Cmin, Cmax = np.nanpercentile(C, pmin), np.nanpercentile(C, pmax)
            # calc color intensities
            RGB.append(im_max*(C - Cmin)/(Cmax - Cmin))
        R, G, B = RGB
        # filters factors
        fR, fG, fB = rgb_f
        # make RGB image
        RGB__yxc = make_lupton_rgb(fR*R, fG*G, fB*B, Q=Q, minimum=minimum, stretch=stretch)
        #################
        #################        
        return RGB__yxc
    
    def source_extractor(self, 
                         sextractor, username, password, 
                         class_star=0.25, satur_level=1600, back_size=64, 
                         detect_thresh=1.1, estimate_fwhm=False,
                         force=False, verbose=0):
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

    def get_iso_sky(self, isophotal_limit=25, isophotal_medsize=10, stars_mask=None, n_sigma=3, n_iter=5, clip_neg=False):
        # Sky selection using rSDSS image
        reference_mag_img__yx = self.mag__lyx[self.filters.index('rSDSS')]
        flux__lyx = self.flux__lyx
        return get_iso_sky(
            refmag__yx=reference_mag_img__yx, flux__lyx=flux__lyx, 
            isophotal_limit=isophotal_limit, isophotal_medsize=isophotal_medsize, 
            stars_mask=stars_mask, n_sigma=n_sigma, n_iter=n_iter, clip_neg=clip_neg,
        )

    def mask_optimal(self):
        f = self._hdulist['DATA'].data
        ef = self._hdulist['ERRORS'].data
        wei = self.weimask__lyx
        return np.bitwise_or(wei>0, f<=0, ~(np.isfinite(ef)))

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