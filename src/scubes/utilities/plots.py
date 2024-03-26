import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib import pyplot as plt

from .io import print_level
from .splusdata import get_lupton_rgb

def plot_mask(detection_image, lupton_rgb, masked_ddata, resulting_mask, sewregions, daoregions=None, save_fig=False, prefix_filename=None, fig=None):
    '''
    Plot a mosaic showing various images and regions related to source detection and masking.

    Parameters
    ----------
    detection_image : str
        The path to the FITS file containing the detection image.

    lupton_rgb : array-like
        The RGB image array used for plotting.

    masked_ddata : array-like
        The masked detection image data.

    resulting_mask : array-like
        The resulting mask array.

    sewregions : list of CirclePixelRegion
        List of regions around sources detected by SExtractor.

    daoregions : list of CirclePixelRegion, optional
        List of regions around sources detected by DAOStarFinder. Defaults to None.

    save_fig : bool, optional
        If True, save the figure as an image file. Defaults to False.

    prefix_filename : str, optional
        The prefix for the saved figure filename. Defaults to 'OBJECT'.
        
    fig : matplotlib.figure.Figure, optional
        The existing figure to use. If None, a new figure will be created. Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure if save_fig is False, otherwise None.
    '''    
    prefix_filename = 'OBJECT' if prefix_filename is None else prefix_filename
    
    dhdu = fits.open(detection_image)
    ddata = dhdu[1].data
    dheader = dhdu[1].header
    wcs = WCS(dheader)
    # FIGURE
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.ion()
    if fig is None:
        fig = plt.figure()
    # AX1
    ax1 = plt.subplot(221, projection=wcs)

    ax1.imshow(lupton_rgb, origin='lower')
    # r_circ.plot(color='y', lw=1.5)
    for sregion in sewregions:
        sregion.plot(ax=ax1, color='g')
    ax1.set_title('RGB')
    # AX2
    ax2 = plt.subplot(222, projection=wcs)
    ax2.imshow(ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
    # r_circ.plot(color='y', lw=1.5)
    for n, sregion in enumerate(sewregions):
        sregion.plot(ax=ax2, color='g')
        ax2.annotate(repr(n), (sregion.center.x, sregion.center.y), color='green')
    if daoregions is not None:
        for dregion in daoregions:
            dregion.plot(ax=ax2, color='m')
    ax2.set_title('Detection')
    # AX3
    ax3 = plt.subplot(223, projection=wcs)
    stars_mask = np.ones(ddata.shape)
    for n, sregion in enumerate(sewregions):
        sregion.plot(ax=ax3, color='g')
    ax3.imshow(masked_ddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
    # r_circ.plot(color='y', lw=1.5)
    ax3.set_title('Masked')
    # AX4
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
        fig_filename = f'{prefix_filename}_maskMosaic.png'
        print_level(f'Saving fig to {fig_filename}')
        fig.savefig(fig_filename, format='png', dpi=180)
        plt.close(fig)
        fig = None
    return fig     
