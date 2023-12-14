from .utilities.io import convert_coord_to_degrees, print_level

class control:
    def __init__(self, args):
        for key, value in args.__dict__.items():
            print_level(f'control obj - key: {key} - value: {value}', 3, args.verbose)
            setattr(self, key, value)
        self._parse_coords()
        self.prefix_filename = f'{self.galaxy}_{self.tile}_{self.size}x{self.size}'

    def _parse_coords(self):
        self.ra, self.dec = convert_coord_to_degrees(self.ra, self.dec)