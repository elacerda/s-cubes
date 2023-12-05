from os import makedirs
from os.path import join

from scubes.utilities.io import check_units, print_level
from scubes.utilities.constants import WAVE_EFF

class control:
    def __init__(self, args, program_name='SCUBES'):
        for key, value in args.__dict__.items():
            print_level(f'control obj - key: {key} - value: {value}', 3, args.verbose)
            setattr(self, key, value)
        self.program_name = program_name
        self._correct_dirpath()
        self._parse_coords()
        self.prefix_filename = f'{self.galaxy}_{self.tile}_{self.size}x{self.size}'
        print_level(f'prefix_filename: {self.prefix_filename}', 2, self.verbose)
        self._make_output_dir()

    def _correct_dirpath(self):
        self.zpcorr_dir = join(self.data_dir, self.zpcorr_dir)
        self.zp_table = join(self.data_dir, self.zp_table)
        self.output_dir = join(self.work_dir, self.galaxy)
        print_level(f'data_dir: {self.data_dir}', 2, self.verbose)
        print_level(f'zpcorr_dir: {self.zpcorr_dir}', 2, self.verbose)
        print_level(f'zp_table: {self.zp_table}', 2, self.verbose)
        print_level(f'output_dir: {self.output_dir}', 2, self.verbose)

    def _parse_coords(self):
        self.ra, self.dec = check_units(self.ra, self.dec)

    def _make_output_dir(self):
        try: 
            makedirs(self.output_dir)
        except FileExistsError:
            print_level(f'{self.program_name}: {self.output_dir}: directory already exists', 2, self.verbose)