import sys
import numpy as np
from astropy.io import fits
from copy import deepcopy as copy
from astropy.visualization import make_lupton_rgb

from .io import print_level
from ..constants import METADATA_NAMES

class read_scube:
    def __init__(self, filename):
        self.filename = filename
        self._read()
        self.init()
    
    def _read(self):       
        try:
            self._hdulist = fits.open(self.filename)
        except FileNotFoundError:
            print_level(f'{self.filename} - file not found')
            sys.exit()

    def init(self):
        self.cent_pix = int(self.size/2)
        
        # MAG
        pixscale = self.data_header.get('PIXSCALE', 0.55)
        a = 1/(2.997925e18*3631.0e-23*pixscale**2)
        x = a*(self.flux__lyx*self.pivot_wave[:, np.newaxis, np.newaxis]**2)
        self.mag__lyx = -2.5*np.ma.log10(x)
        self.emag__lyx = (2.5*np.ma.log10(np.exp(1)))*self.eflux__lyx/self.flux__lyx

    def lRGB_image(
        self, rgb=('rSDSS', 'gSDSS', 'iSDSS'), rgb_f=(1, 1, 1), 
        pminmax=(5, 95), im_max=255, 
        # astropy.visualization.make_lupton_rgb() input vars
        minimum=(0, 0, 0), Q=0, stretch=10):
        '''
        make RGB
        '''
        # create filters indexes list
        def _parse_filters(f_tup):
            if isinstance(f_tup, list):
                f_tup = tuple(f_tup)
            i_f = []
            if isinstance(f_tup, tuple):
                for f in f_tup:
                    if isinstance(f, str):
                        i_f.append(self.filters.index(f))
                    else:
                        i_f.append(f)
            else:
                if isinstance(f_tup, str):
                    i_f.append(self.filters.index(f_tup))
                else:
                    i_f.append(f_tup)
            return i_f
        
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
        i_r = _parse_filters(rgb[0])
        i_g = _parse_filters(rgb[1])
        i_b = _parse_filters(rgb[2])
        # get filters fluxes
        R = copy(self.flux__lyx[i_r, :, :].filled(np.nan)).sum(axis=0)
        G = copy(self.flux__lyx[i_g, :, :].filled(np.nan)).sum(axis=0)
        B = copy(self.flux__lyx[i_b, :, :].filled(np.nan)).sum(axis=0)
        # percentiles
        pmin, pmax = pminmax
        Rmin, Rmax = np.nanpercentile(R, pminmax[0]), np.nanpercentile(R, pminmax[1])
        Gmin, Gmax = np.nanpercentile(G, pminmax[0]), np.nanpercentile(G, pminmax[1])
        Bmin, Bmax = np.nanpercentile(B, pminmax[0]), np.nanpercentile(B, pminmax[1])
        # R, G and B images
        R = im_max*(R - Rmin)/(Rmax - Rmin)
        G = im_max*(G - Gmin)/(Gmax - Gmin)
        B = im_max*(B - Bmin)/(Bmax - Bmin)
        # filters factors
        fR, fG, fB = rgb_f
        # make RGB image
        RGB__yx = make_lupton_rgb(fR*R, fG*G, fB*B, Q=Q, minimum=minimum, stretch=stretch)
        #################
        #################        
        return RGB__yx

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
    def x0tile(self):
        return self.primary_header.get('X0TILE', None)
    
    @property
    def y0tile(self):
        return self._hdulist[0].header['Y0TILE']
    
    @property 
    def weimask__yx(self):
        return self._hdulist['WEIMASK'].data

    @property 
    def flux__lyx(self):
        return np.ma.masked_array(self._hdulist['DATA'].data, mask=self.weimask__lyx>0, copy=True)

    @property 
    def eflux__lyx(self):
        return np.ma.masked_array(self._hdulist['ERRORS'].data, mask=self.weimask__lyx>0, copy=True)