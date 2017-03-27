"""
tdoa.py

TDOA (time difference of arrival) estimation.

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
from scipy import stats

from .cc import pairwise_cc

def _tdoa_shift(idx, cc_size, fs=None, upsample=1):
    """convert index in cc to tdoa"""
    if idx > cc_size / 2:
        idx = idx - cc_size 
    if fs is None:
        return idx
    else:
        return 1.0 * idx / (fs * upsample)

def single_tdoa(x, y, cc_func, fs=None, upsample=1):
    """Estimate TDOA by finding peak in (G)CC.

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        cc_func  : cross correlation function.
        fs       : sample rate, if not given (default) the result is number
                   of samples (after upsampling).
        upsample : an integer indicating factor of upsampling.
                   default value is 1.

    Returns:
        tdoa    : estimate of TDOA
    """
    cc = cc_func(x, y, upsample)
    return _tdoa_shift(np.argmax(cc), len(cc), fs, upsample)

def tdoa_hist(cc_atime, fs=None, upsample=1):
    """Estimate TDOA by accumulating histograms of TDOA estimates across
       time.

    Args:
        cc_atime : cross correlation between two signals across time
        fs       : sample rate, if not given (default) the result is number
                   of samples (after upsampling).
        upsample : an integer indicating factor of upsampling.
                   default value is 1.

    Returns:
        tdoa    : estimate of TDOA
    """
    idx, _ = stats.mode(np.argmax(cc_atime, axis=1))
    return _tdoa_shift(idx, cc_atime.shape[1], fs, upsample)

def tdoa_sum(cc_atime, fs=None, upsample=1):
    """Estimate TDOA by find peak in the sum of cc.

    Args:
        cc_atime : cross correlation between two signals across time
        fs       : sample rate, if not given (default) the result is number
                   of samples (after upsampling).
        upsample : an integer indicating factor of upsampling.
                   default value is 1.

    Returns:
        tdoa    : estimate of TDOA
    """
    cc_sum = np.sum(cc_atime, axis=0)
    return _tdoa_shift(np.argmax(cc_sum), cc_atime.shape[1], fs, upsample)

def pairwise_tdoa(tf, cc_func, tdoa_func, fs=None, upsample=1):
    """Estimation TDOA between all channel pairs.

    The TDOA is estimated by finding the peak of cross-correlation summed
    across time.
    
    Args:
        tf        : multi-channel time-frequency domain signal.
        cc_func   : cross correlation function.
        tdoa_func : tdoa estimation function.
        fs        : sample rate, if not given (default) the result is number
                    of samples (after upsampling).
        upsample  : an integer indicating factor of upsampling.
                    default value is 1.

    Returns:
        pw_tdoa   : TDOA between all channel pairs.
                    dict : (channel id, channel id) -> tdoa estimate.
    """
    pw_cc = pairwise_cc(tf, cc_func, upsample)
    return {k : tdoa_func(cc, fs, upsample) for k, cc in pw_cc.items()}

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

