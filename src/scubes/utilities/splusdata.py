from PIL import Image
from tqdm import tqdm
from numpy import array
from splusdata import Core
from astropy.io import fits
from os.path import join, isfile
from splusdata.core import AuthenticationError

from .io import print_level

def connect_splus_cloud(username=None, password=None):
    n_tries = 0
    conn = None
    while (n_tries < 3) and (conn is None):
        try:
            conn = Core(username=username, password=password)
        except AuthenticationError:
            n_tries += 1
    return conn

def download_splus_tiles(conn, tile, bands, output_dir=None, download_weight=True, overwrite=False):
    fnames = []
    if output_dir is None:
        output_dir = '.'
    for filt in tqdm(bands, desc=f'{tile} - downloading', leave=False, position=1):
        fname = join(output_dir, f'{tile}_{filt}_swp.fits.fz')
        fnames.append(fname)
        if not isfile(fname) or overwrite:
            t = conn.field_frame(field=tile, band=filt, filename=fname)
        if download_weight:
            fname = join(output_dir, f'{tile}_{filt}_swpweight.fits.fz')
            fnames.append(fname)
            if not isfile(fname):
                wt = conn.field_frame(field=tile, band=filt, weight=True, filename=fname)
    return fnames

def download_splus_stamps(conn, ra, dec, size, tile, obj_name, bands, output_dir=None, download_weight=True, overwrite=False):
    fnames = []
    if output_dir is None:
        output_dir = '.'
    for filt in tqdm(bands, desc=f'{obj_name} @ {tile} - downloading', leave=False, position=1):
        fname = join(output_dir, f'{obj_name}_{tile}_{filt}_{size}x{size}_swp.fits.fz')
        fnames.append(fname)
        if not isfile(fname) or overwrite:
            _ = conn.stamp(ra=ra, dec=dec, size=size, band=filt, weight=False, option=tile, filename=fname)
        if download_weight:
            fname = join(output_dir, f'{obj_name}_{tile}_{filt}_{size}x{size}_swpweight.fits.fz')
            fnames.append(fname)
            if not isfile(fname) or overwrite:
                _ = conn.stamp(ra=ra, dec=dec, size=size, band=filt, weight=True, option=tile, filename=fname)
    return fnames

def download_splus_detection_image(conn, ra, dec, size, tile, obj_name, output_dir=None, band=None, overwrite=False):
    band = 'G,R,I,Z' if band is None else band
    if output_dir is None:
        output_dir = '.' 
    fname = join(output_dir, f'{obj_name}_{tile}_{size}x{size}_detection.fits')
    if not isfile(fname) or overwrite:
        print_level(f' {obj_name} @ {tile} - downloading detection image')
        t = conn.stamp_detection(ra=ra, dec=dec, size=size, bands=band, option=tile)  #, filename=join(output_dir, fname))
        phdu = fits.PrimaryHDU()
        ihdu = fits.ImageHDU(data=t[1].data, header=t[1].header, name='IMAGE')
        hdul = fits.HDUList([phdu, ihdu])
        hdul.writeto(fname, overwrite=overwrite)
    return fname

def download_splus_lupton_rgb(conn, ra, dec, size, tile, obj_name, output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = '.'
    fname = join(output_dir, f'{obj_name}_{tile}_{size}x{size}.png')
    if not isfile(fname) or overwrite:
        print_level(f'{obj_name} @ {tile} - downloading RGB image')
        img = conn.lupton_rgb(ra=ra, dec=dec, size=size, option=tile) 
        img.save(fname, 'PNG')
    else:
        img = Image.open(fname)
    return img.transpose(Image.FLIP_TOP_BOTTOM)
