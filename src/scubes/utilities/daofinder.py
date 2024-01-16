import numpy as np
from photutils import DAOStarFinder
from regions import PixCoord, CirclePixelRegion

from .io import print_level

def DAOfinder(data, mean=0, median=0, std=0.5):
    '''
    Calculate photometry using DAOStarFinder.

    Parameters
    ----------
    data : array-like
        The 2D array from which to extract sources.

    mean : float, optional
        The mean background value. Defaults to 0.

    median : float, optional
        The median background value. Defaults to 0.
        
    std : float, optional
        The standard deviation of the background. Defaults to 0.5.

    Returns
    -------
    Table
        A table containing the found sources and their properties.
    '''    
    # DETECT TOO MUCH HII REGIONS
    print_level(('mean', 'median', 'std'))
    print_level((mean, median, std))
    print_level('Running DAOfinder...')
    daofind = DAOStarFinder(fwhm=4.0, sharplo=0.2, sharphi=0.9, roundlo=-0.5, roundhi=0.5, threshold=5.*std)
    sources = daofind(data)
    return sources

def DAOregions(data):
    '''
    Extract circular regions around sources found by DAOfinder.

    Parameters
    ----------
    data : array-like
        The 2D array from which to extract sources.

    Returns
    -------
    list
        List of CirclePixelRegion objects representing the regions around the detected sources.
    '''    
    daocat = DAOfinder(data=data)
    daopos = np.transpose((daocat['xcentroid'], daocat['ycentroid']))
    daorad = 4*(abs(daocat['sharpness']) + abs(daocat['roundness1']) + abs(daocat['roundness2']))
    return [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(daopos, daorad)]
