#!/usr/bin/env python3
from os import listdir, walk
from os.path import join, isdir
from distutils.core import setup

def version(fn):
    v = ''
    with open(fn, 'r') as f:
        for l in f.readlines():
            if '__version__' in l:
                v = l.split('=')[-1].strip().replace("'", '').split(' ')[-1][1:]
    return v

DATA_DIRNAME='data'
SCRIPTS_DIRNAME = 'bin'
VERSION_FILE = 'scubes/utilities/constants.py'

all_packages = ['scubes', 'scubes.utilities']
packages_data = {
    package: [f'{DATA_DIRNAME}/*']+[f'{join(DATA_DIRNAME, sub)}/*' for root, subs, files in walk(join(package, DATA_DIRNAME)) for sub in subs]
    for package in all_packages if isdir(join(package, DATA_DIRNAME))
}
scripts = [
    join(SCRIPTS_DIRNAME, script_name)
    for script_name in listdir(SCRIPTS_DIRNAME) if script_name.endswith('.py')
]
version = version(VERSION_FILE)

requirements = [
    'pandas',
    'matplotlib',
    'numpy',
    'scipy',
    'astropy',
    'regions',
    'tqdm',
    'photutils',
    'splusdata',
]

setup(
    name='S-Cubes',
    version=version,
    description='Make galaxy cubes (X, Y, Lambda) with S-PLUS data.',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    keywords='galaxies',
    url='https://github.com/elacerda/s-cubes',
    download_url='https://github.com/elacerda/s-cubes/archive/refs/heads/main.zip',
    author='Eduardo Alberto Duarte Lacerda',
    author_email='dhubax@gmail.com',
    license='GPLv3',
    packages=all_packages,
    setup_requires=['wheel'],
    install_requires=requirements,
    include_package_data=True,
    package_data=packages_data,
    scripts=scripts,
)