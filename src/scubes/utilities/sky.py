import numpy as np
from scipy.ndimage import median_filter

def get_iso_sky(refmag__yx, flux__lyx, isophotal_limit=25, isophotal_medsize=10, stars_mask=None, n_sigma=3, n_iter=5, clip_neg=False):
    sky_mask__yx = ~(refmag__yx <= isophotal_limit) & (stars_mask == 0)
    # from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
    # size gives the shape that is taken from the input array, 
    # at every element position, to define the input to the filter 
    # function
    sky_pixels__yx = median_filter(sky_mask__yx, size=isophotal_medsize) | (stars_mask == 1)
    
    # clipping sky fluxes for outliers
    sky_flux__lyx = np.ma.zeros_like(flux__lyx)
    nl, ny, nx = sky_flux__lyx.shape

    for i in range(nl):
        flux_filt__yx = flux__lyx[i]
        sky_flux__yx = np.ma.masked_array(flux_filt__yx, mask=~sky_pixels__yx)
        n = n_iter
        while n > 0:
            _med = np.ma.median(sky_flux__yx)
            _sig = np.ma.std(sky_flux__yx)
            outliers = (sky_flux__yx - _med) > (n_sigma*_sig)
            if clip_neg:
                outliers = np.ma.abs(sky_flux__yx - _med) > (n_sigma*_sig)
            sky_flux__yx[outliers] = np.ma.masked
            n -= 1
        sky_flux__lyx[i] = sky_flux__yx

    sky = {}
    sky['flux__lyx'] = sky_flux__lyx
    sky['mask__yx'] = (np.ma.getmask(sky_flux__lyx).sum(axis=0) == 0)
    sky['mean__l'] = np.ma.mean(sky_flux__lyx, axis=(1, 2))
    sky['median__l'] = np.ma.median(sky_flux__lyx, axis=(1, 2))
    sky['std__l'] = np.ma.std(sky_flux__lyx, axis=(1, 2))

    return sky

