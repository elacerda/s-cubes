from .utilities.io import convert_coord_to_degrees, print_level

class control:
    '''
    Class for handling control parameters and configurations.

    Parameters
    ----------
    args : argparse.ArgumentParser
        An object containing arguments for control.

    Attributes
    ----------
    ra : float
        Right Ascension coordinate.
    
    dec : float
        Declination coordinate.
    
    galaxy : str
        Galaxy identifier.
    
    tile : str
        Tile identifier.
    
    size : int
        Size of the image.
    
    verbose : int
        Verbosity level for printing debug information.
    
    prefix_filename : str
        Filename prefix based on galaxy, tile, and size.

    Methods
    -------
    _parse_coords()
        Convert the input right ascension and declination to degrees.
    '''    
    def __init__(self, args):
        for key, value in args.__dict__.items():
            print_level(f'control obj - key: {key} - value: {value}', 3, args.verbose)
            setattr(self, key, value)
        self._parse_coords()
        self.prefix_filename = f'{self.galaxy}_{self.tile}_{self.size}x{self.size}'

    def _parse_coords(self):
        '''
        Convert the input right ascension and declination to degrees.
        '''        
        self.ra, self.dec = convert_coord_to_degrees(self.ra, self.dec)