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
    'mar': MAR_KEYS,
}

AUTHORS = list(KEYS.keys())

def get_keys(author):
    '''
    Get the dictionary of keys associated with a specific author.

    Parameters
    ----------
    author : str
        Author identifier.

    Returns
    -------
    dict or None
        Dictionary of keys for the specified author or None if not found.
    '''
    keys = KEYS.get(author, None)
    if keys is None:
        Warning(f'{author}: missing author, using jype dictionary')
        keys = KEYS['jype']
    return keys

def get_key(ikey, author):
    '''
    Get the key associated with a specific keyword for a given author.

    Parameters
    ----------
    ikey : str
        Keyword to retrieve.
    
    author : str
        Author identifier.

    Returns
    -------
    str
        Key associated with the keyword for the specified author.
    '''    
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
    '''
    Get the author associated with a given header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header.

    Returns
    -------
    str
        Author identifier.
    '''   
    author = header.get('AUTHOR', None)
    if author is None:
        Warning(f'{author}: missing author, using jype dictionary')
        author = 'mar'
    return author
