#!/usr/bin/env python3
import sys
from shutil import which
from os.path import basename, dirname, realpath

__script_name__ = basename(sys.argv[0])

from scubes.utilities.io import print_level
from scubes.utilities.args import create_parser
from scubes.core import SCubes

def parse_arguments():
    parser = create_parser()
    if len(sys.argv) == 1:
        print_level(f'{__script_name__}: missing arguments', 0, 1)
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(args=sys.argv[1:])
    _sex = which(args.sextractor)
    if _sex is None:
        print_level(f'{__script_name__}: {args.sextractor}: SExtractor exec not found', 2, args.verbose)
        _SExtr_names = ['sex', 'source-extractor']
        for name in _SExtr_names:
            _sex = which(name)
            if _sex is None:
                print_level(f'{__script_name__}: {name}: SExtractor exec not found', 2, args.verbose)
            else:
                args.sextractor = _sex
                pass
        if _sex is None:
            print_level(f'{__script_name__}: SExtractor not found')
            sys.exit(1)
    return args

if __name__ == '__main__':
    path = dirname(realpath(__file__))
    program_name = __script_name__
    scubes = SCubes(args=parse_arguments(), path=path, program_name=program_name)
    scubes.make(get_mask=True, det_img=True, flam_scale=None)