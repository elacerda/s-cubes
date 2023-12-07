import sys
from shutil import which
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from .io import print_level

class readFileArgumentParser(ArgumentParser):
    '''
    A class that extends :class:`argparse.ArgumentParser` to read arguments from file.

    Methods
    -------
    convert_arg_line_to_args: yeld's arguments reading a file.
    '''

    def __init__(self, *args, **kwargs):
        super(readFileArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

def create_parser(args_dict, program_description=None):
    '''
    Create the parser for the arguments of the `scubes` entry-point console script.
    It uses two constants defined at `scubes.utilities.constants.py`. The program 
    description `SPLUS_PROG_DESC` and the  dictionary of arguments `SPLUS_ARGS`.

    It uses :class:`readFileArgumentParser` to include the option to read the 
    arguments from a file using the prefix @ before the filename::

        scubes @file.args

    Parameters
    ----------
    args_dict : dict
        Dictionary to with the long options as keys and a list containing the short 
        option char at the first position and the `kwargs` for the argument 
        configuration.

    program_description : str, optional
        Program description. Default is None.

    Returns
    -------
    parser : :class:`argparse.ArgumentParser`
        The arguments parser.

    See Also
    --------
    `argparse.ArgumentParser.add_argument()`
    '''
    _formatter = lambda prog: RawDescriptionHelpFormatter(prog, max_help_position=30)
    
    parser = readFileArgumentParser(
        fromfile_prefix_chars='@', 
        description=program_description, 
        formatter_class=_formatter
    )
    for k, v in args_dict.items():
        long_option = k
        short_option, kwargs = v
        option_string = []
        positional = False
        if short_option != '':
            if short_option == 'pos':
                positional = True
            else:
                option_string.append(f'-{short_option}')
        if positional:
            option_string.append(long_option)
        else:
            option_string.append(f'--{long_option}')
        if (long_option != 'verbose') and (kwargs.get('default') is not None):
            _tmp = kwargs['help']
            kwargs['help'] = f'{_tmp} Default value is %(default)s'
        parser.add_argument(*option_string, **kwargs)
    return parser

def parse_arguments(argv, parser):
    '''
    Parse the command-line arguments with `parser` in `argv` list of arguments.

    Parameters
    ----------
    argv : list
        List of arguments such as `sys.argv`.

    parser : :class:`argparse.ArgumentParser`
        Parser to `argv` command-line arguments.

    Returns
    -------
    args : :class:`argparse.Namespace`
        The arguments namespace class attributes.
    '''
    # CREATE PARSER
    args = parser.parse_args(args=argv[1:])
    _sex = which(args.sextractor)
    if _sex is None:
        print_level(f'{args.sextractor}: SExtractor exec not found', 1, args.verbose)
        _SExtr_names = ['sex', 'source-extractor']
        for name in _SExtr_names:
            _sex = which(name)
            if _sex is None:
                print_level(f'{name}: SExtractor exec not found', 2, args.verbose)
            else:
                print_level(f'{name}: SExtractor found. Forcing --sextractor={_sex}', 1, args.verbose)
                args.sextractor = _sex
                pass
        if _sex is None:
            print_level(f'SExtractor not found')
            sys.exit(1)
    return args