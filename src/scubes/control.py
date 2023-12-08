from os import makedirs
from os.path import join

from .utilities.io import check_units, print_level

class control:
    def __init__(self, args):
        for key, value in args.__dict__.items():
            print_level(f'control obj - key: {key} - value: {value}', 3, args.verbose)
            setattr(self, key, value)
        self.output_dir = join(self.work_dir, self.galaxy)
        self._parse_coords()
        self.prefix_filename = f'{self.galaxy}_{self.tile}_{self.size}x{self.size}'
        print_level(f'output_dir: {self.output_dir}', 2, self.verbose)
        print_level(f'prefix_filename: {self.prefix_filename}', 2, self.verbose)
        self._make_output_dir()

    def _parse_coords(self):
        self.ra, self.dec = check_units(self.ra, self.dec)

    def _make_output_dir(self):
        try: 
            makedirs(self.output_dir)
        except FileExistsError:
            print_level(f'{self.output_dir}: directory already exists', 2, self.verbose)