__version__ = 'v0.1.0-beta'

__author__ = 'Eduardo A. D. Lacerda'

__author_email__ = 'dhubax@gmail.com'

DATA_DIR = 'data'
ZPCORR_DIR = 'zpcorr_iDR4'
ZP_TABLE = 'iDR4_zero-points.csv'

MOTD_TOP = '┌─┐   ┌─┐┬ ┬┌┐ ┌─┐┌─┐ '
MOTD_MID = '└─┐───│  │ │├┴┐├┤ └─┐ '
MOTD_BOT = '└─┘   └─┘└─┘└─┘└─┘└─┘ '
MOTD_SEP = '----------------------'

PROG_DESC = f'''
{MOTD_TOP} | Create S-PLUS galaxies data cubes, a.k.a. S-CUBES. 
{MOTD_MID} | S-CUBES is an organized FITS file with data, errors, 
{MOTD_BOT} | mask and metadata about some galaxy present on any 
{MOTD_SEP} + S-PLUS observed tile. Any problem contact:

                {__author__} - {__author_email__}
'''

WAVE_EFF = {
    'U': 3536.0,
    'F378': 3770.0,
    'F395': 3940.0,
    'F410': 4094.0,
    'F430': 4292.0,
    'G': 4751.0,
    'F515': 5133.0,
    'R': 6258.0,
    'F660': 6614.0,
    'I': 7690.0,
    'F861': 8611.0,
    'Z': 8831.0,
}

EXPTIMES = {
    'F378': 660, 'F395': 354, 'F410': 177, 'F430': 171,  
    'F515': 183, 'F660': 870, 'F861': 240,
    'U': 681, 'G': 99, 'R': 120, 'I': 138, 'Z': 168,
}

NAMES_CORRESPONDENT = {
    'F378': 'J0378', 'F395': 'J0395','F410': 'J0410', 'F430': 'J0430', 
    'F515': 'J0515', 'F660': 'J0660', 'F861': 'J0861', 
    'U': 'u', 'G': 'g', 'R': 'r', 'I': 'i', 'Z': 'z',
}
