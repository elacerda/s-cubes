import numpy as np

def robustStat(arr, sigma=3, iters=5, med=False):
    '''
    Computes various robust statistics for an input array.

    Parameters
    ----------
    arr : numpy array
        The input array.

    sigma : float, optional
        The number of standard deviations allowed for data to be considered as an outlier.
        Defaults to 3.
        
    iters : int, optional
        The number of iterations to perform the outlier rejection process. Defaults to 5.

    med : bool, optional
        If True, use the median absolute deviation (MAD) instead of the standard deviation for
        outlier rejection. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the maximum, minimum, mean, median, standard deviation,
        fraction of rejected data points, robust standard deviation (nmad), and robust
        standard deviation (rms) of the input array.
    '''
    from astropy.stats import mad_std

    arr = np.array(list(filter(lambda v: v==v, arr)))
    arr = arr[arr != np.array(None)]
    
    rms = None
    good = np.ones(len(arr), dtype=int)
    nx = sum(good)

    for i in range(len(arr)):
        if i > 0: xs = np.compress(good, arr)
        else: xs = arr
        aver = np.median(xs)
        if med: rms = mad_std(xs)
        else: rms = np.std(xs)
        good = good * np.less_equal(abs(arr - aver), sigma * rms)
        nnx = sum(good)
        if nnx == nx: break
        else: nx = nnx

    remaining = np.compress(good, arr)
    n_remaining = len(remaining)
    
    if n_remaining > 3:
        maxi = max(remaining)
        mini = min(remaining)
        mean = np.mean(remaining, dtype='float64')
        std = np.std(remaining, dtype='float64')
        median = np.median(remaining)
        auxl = float(len(remaining)) / 3.
        auxsort = remaining * 1.0
        auxsort.sort()

        fraction = 1. - (float(n_remaining) / float(len(arr)))
        nmad = mad_std(remaining)

    elif n_remaining > 1:
        maxi = max(remaining)
        mini = min(remaining)
        mean = np.mean(remaining, dtype='float64')
        std = np.std(remaining, dtype='float64')
        median = np.median(remaining)
        fraction = 1. - (float(n_remaining) / float(len(arr)))
        nmad = mad_std(remaining)
    elif n_remaining > 0:
        maxi = max(remaining)
        mini = min(remaining)
        mean = np.mean(remaining, dtype='float64')
        median = np.median(remaining)
        std = -1.0
        fraction = 1. - (float(n_remaining) / float(len(arr)))
        nmad = -99.99
    else:
        maxi = -1.0
        mini = 0.0
        mean = -1.0
        median = -1.0
        std = -1.0
        fraction = -1
        nmad = -99.99
    
    return {'maxi' :maxi, 'mini': mini, 'mean': mean, 'median': median, 'std':std, 'fraction': fraction, 'nmad':nmad, 'rms': rms}





   