from astropy.io import fits
from splusdata import Core
from splusdata.core import AuthenticationError

def connect_splus_cloud(username=None, password=None):
    '''
    Connect to the S-PLUS Cloud service.

    Parameters
    ----------
    username : str, optional
        The username for S-PLUS Cloud authentication.

    password : str, optional
        The password for S-PLUS Cloud authentication.

    Returns
    -------
    splusdata.Core or None
        An instance of the S-PLUS Cloud connection (`splusdata.Core`) or
        `None` if the authentication fails after three attempts.
    '''    
    n_tries = 0
    conn = None
    while (n_tries < 3) and (conn is None):
        try:
            conn = Core(username=username, password=password)
        except AuthenticationError:
            n_tries += 1
    return conn

def detection_image_hdul(conn, wcs=False, **kwargs):
    '''
    Retrieve the detection image from the S-PLUS Cloud service.

    Parameters
    ----------
    conn : :class:`splusdata.Core`
        An instance of the S-PLUS Cloud connection.

    wcs : bool, optional
        If True, update the header with World Coordinate System (WCS)
        information. Default is False.

    \**kwargs : dict
        Additional keyword arguments to be passed to :meth:`conn.stamp_detection`.

    Returns
    -------
    :class:`astropy.io.fits.HDUList`
        An HDUList containing a PrimaryHDU and an ImageHDU representing the
        detection image. If `wcs` is True, the header of the ImageHDU is
        updated with WCS information.
    '''
    kwargs['_data_release'] = 'dr4'
    t = conn.stamp_detection(**kwargs) 
    phdu = fits.PrimaryHDU()
    # UPDATE HEADER WCS
    if wcs:
        from astropy.wcs import WCS

        w = WCS(t[1].header)
        t[1].header.update(w.to_header())
    ihdu = fits.ImageHDU(data=t[1].data, header=t[1].header, name='IMAGE')
    hdul = fits.HDUList([phdu, ihdu])
    return hdul
    
def get_lupton_rgb(conn, transpose=False, save_img=False, filename=None, **kwargs):
    '''
    Retrieve an RGB image using the Lupton algorithm from the S-PLUS Cloud service.

    Parameters
    ----------
    conn : splusdata.Core
        An instance of the S-PLUS Cloud connection.

    transpose : bool, optional
        If True, transpose the RGB image. Default is False.

    save_img : bool, optional
        If True, save the RGB image to a file. Default is False.

    filename : str, optional
        The filename to save the RGB image if `save_img` is True. If None,
        the default filename is 'OBJECT.png'.

    \**kwargs : dict
        Additional keyword arguments to be passed to :meth:`conn.lupton_rgb`.

    Returns
    -------
    :class:`PIL.Image.Image`
        An RGB image represented as a PIL Image object.
    '''
    from PIL import Image

    img = conn.lupton_rgb(**kwargs) 
    img = img.transpose(Image.FLIP_TOP_BOTTOM) if transpose else img
    if save_img:
        fname = 'OBJECT.png' if filename is None else filename
        img.save(fname, 'PNG')
    return img