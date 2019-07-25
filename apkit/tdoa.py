"""
tdoa.py

TDOA (time difference of arrival) estimation.

TDOA between x and y means the y is delayed by TDOA wrt x.

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
from scipy import stats

def _tdoa_shift(idx, cc_size, fs=None):
    """convert index in cc to tdoa"""
    if idx > cc_size / 2:
        idx = idx - cc_size 
    if fs is None:
        return idx
    else:
        return 1.0 * idx / fs

def single_tdoa(cc, fs=None):
    """Estimate TDOA by finding peak in (G)CC.

    Args:
        cc : cross correlation.
        fs : sample rate (after upsampling),
             if None (default) the result is number of samples.

    Returns:
        tdoa    : estimate of TDOA
    """
    return _tdoa_shift(np.argmax(cc), len(cc), fs)

def tdoa_hist(cc_atime, fs=None):
    """Estimate TDOA by accumulating histograms of TDOA estimates across
       time.

    Args:
        cc_atime : cross correlation between two signals across time
        fs       : sample rate (after upsampling),
                   if None (default) the result is number of samples.

    Returns:
        tdoa    : estimate of TDOA
    """
    [idx], _ = stats.mode(np.argmax(cc_atime, axis=1))
    return _tdoa_shift(idx, cc_atime.shape[1], fs)

def tdoa_sum(cc_atime, fs=None):
    """Estimate TDOA by find peak in the sum of cc.

    Args:
        cc_atime : cross correlation between two signals across time
        fs       : sample rate (after upsampling),
                   if None (default) the result is number of samples.

    Returns:
        tdoa    : estimate of TDOA
    """
    cc_sum = np.sum(cc_atime, axis=0)
    return _tdoa_shift(np.argmax(cc_sum), cc_atime.shape[1], fs)

def pairwise_tdoa(pw_cc, tdoa_func, fs=None):
    """Estimation TDOA between all channel pairs.

    The TDOA is estimated by finding the peak of cross-correlation summed
    across time.
    
    Args:
        pw_cc     : pairwise cross correlation
        tdoa_func : tdoa estimation function.
        fs        : sample rate (after upsampling),
                    if None (default) the result is number of samples.

    Returns:
        pw_tdoa   : TDOA between all channel pairs.
                    dict : (channel id, channel id) -> tdoa estimate.
    """
    return {k: tdoa_func(cc, fs) for k, cc in pw_cc.iteritems()}

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

