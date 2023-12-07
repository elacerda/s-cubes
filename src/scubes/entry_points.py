import sys

from .utilities.args import create_parser, parse_arguments
from .constants import SPLUS_ARGS, SPLUS_PROG_DESC


def scubes():
    from .core import SCubes

    parser = create_parser(args_dict=SPLUS_ARGS, program_description=SPLUS_PROG_DESC)
    args = parse_arguments(argv=sys.argv, parser=parser)

    scubes = SCubes(args)
    scubes.make(get_mask=True, det_img=True, flam_scale=None)