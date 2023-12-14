import numpy as np
from photutils import DAOStarFinder
from regions import PixCoord, CirclePixelRegion

from .io import print_level

def DAOfinder(data, mean=0, median=0, std=0.5):
    "calculate photometry using DAOfinder"
    # DETECT TOO MUCH HII REGIONS
    print_level(('mean', 'median', 'std'))
    print_level((mean, median, std))
    print_level('Running DAOfinder...')
    daofind = DAOStarFinder(fwhm=4.0, sharplo=0.2, sharphi=0.9, roundlo=-0.5, roundhi=0.5, threshold=5.*std)
    sources = daofind(data)
    return sources

def DAOregions(data):
    daocat = DAOfinder(data=data)
    daopos = np.transpose((daocat['xcentroid'], daocat['ycentroid']))
    daorad = 4*(abs(daocat['sharpness']) + abs(daocat['roundness1']) + abs(daocat['roundness2']))
    return [CirclePixelRegion(center=PixCoord(x, y), radius=z) for (x, y), z in zip(daopos, daorad)]
