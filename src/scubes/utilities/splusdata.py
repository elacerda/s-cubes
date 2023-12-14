from astropy.io import fits
from splusdata import Core
from splusdata.core import AuthenticationError

def connect_splus_cloud(username=None, password=None):
    n_tries = 0
    conn = None
    while (n_tries < 3) and (conn is None):
        try:
            conn = Core(username=username, password=password)
        except AuthenticationError:
            n_tries += 1
    return conn

def detection_image_hdul(conn, wcs=False, **kwargs):
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
    from PIL import Image

    img = conn.lupton_rgb(**kwargs) 
    img = img.transpose(Image.FLIP_TOP_BOTTOM) if transpose else img
    if save_img:
        fname = 'OBJECT.png' if filename is None else filename
        img.save(fname, 'PNG')
    return img
    