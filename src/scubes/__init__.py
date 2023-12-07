from importlib.metadata import version, metadata

scubes_meta = metadata('s-cubes')
__author__ = scubes_meta['Author-email']
__version__ = version('s-cubes')
