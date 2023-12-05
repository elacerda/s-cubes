JYPE_KEYS = {
    'PSFFWHM': 'HIERARCH OAJ PRO FWHMMEAN',
    'EFFTIME': 'EFECTIME',
}

MAR_KEYS = {
    'PSFFWHM': 'HIERARCH MAR PRO FWHMMEAN',
    'EFFTIME': 'EFECTIME',
}

KEYS = {
    'jype': JYPE_KEYS,
    #'mar': MAR_KEYS,
}

AUTHORS = list(KEYS.keys())

def get_keys(author):
    keys = KEYS.get(author, None)
    if keys is None:
        Warning(f'{author}: missing author, using jype dictionary')
        keys = KEYS['jype']
    return keys

def get_key(ikey, author):
    keys = get_keys(author)
    if keys is None:
        Warning(f'{author}: {ikey}: not in author\'s dictionary')
        return ikey
    else:
        key = keys.get(ikey, None)
    if key is None:
        Warning(f'{author}: {ikey}: not in author\'s dictionary')
        key = ikey
    return key

def get_author(header):
    author = header.get('AUTHOR', None)
    if author is None:
        Warning(f'{author}: missing author, using jype dictionary')
        author = 'jype'
    return author
