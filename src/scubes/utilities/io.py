import re
import pandas as pd
from os.path import basename
from datetime import datetime
from astropy.coordinates import SkyCoord

def print_level(msg, level=0, verbose=0):
    import __main__ as main
    try:
        __script_name__ = basename(main.__file__)
    except AttributeError:
        __script_name__ = ''

    if verbose >= level:
        print(f'[{datetime.now().isoformat()}] - {__script_name__}: {msg}')

def check_units(ra, dec):
    # Pattern to match: any letter (a-z, A-Z) or "°"
    # Default unit is degrees, so if no unit, then is assumed as degs. 
    def check_deg(input_string):
        pattern = r'[a-zA-Z°]'
        # re.search returns a match object if the pattern is found, None otherwise
        if re.search(pattern, input_string):
            return input_string
        else:
            return input_string + '°'

    def check(input_string):
        pattern = r'[hdms]'
        x = re.search(pattern, input_string)
        if x is None:
            return check_deg(input_string)
        else:
            return input_string
        
    if not isinstance(ra, str):
        ra = str(ra)

    if not isinstance(dec, str):
        dec = str(dec)

    return check(ra), check(dec)

def convert_coord_to_degrees(ra, dec, frame='icrs'):
    """
    Convert the coordinate to degrees.
    This should be called before any specific function that requires coordinates.
    """
    ra, dec = check_units(ra, dec)

    # Create a SkyCoord object
    c = SkyCoord(ra, dec, frame=frame)

    # Return RA and Dec in degrees
    return c.ra.deg, c.dec.deg

def read_first_line(filename):
    """Return the first record of the given filename

    Parameters
    ----------
    filename : str
        The name of the file for which to get the first record

    Returns
    -------
    str
        The first not commented record found in the given filename
    """
    with open(filename) as f:
        l = f.readline()
        # jump comments
        while l.startswith('#'):
            l = f.readline()
    return l.strip()