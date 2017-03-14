"""
cc.py

(general) cross correlations

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
from itertools import izip

def gcc_phat(x, y, upsample=1):
    """GCC-PHAT

    Args:
        x  : 1-d array, frequency domain signal 1
        y  : 1-d array, frequency domain signal 2
        upsample : an integer indicating factor of upsampling.

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    cpsd = x * y.conj()
    return np.real(np.fft.ifft(cpsd / np.abs(cpsd), n=cpsd.shape[-1] * upsample))

def cross_correlation(x, y, upsample=1):
    """Cross correlation

    Args:
        x  : 1-d array, frequency domain signal 1
        y  : 1-d array, frequency domain signal 2
        upsample : an integer indicating factor of upsampling.

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    cpsd = x * y.conj()
    return np.real(np.fft.ifft(cpsd, n=cpsd.shape[-1] * upsample)
                    / np.max(np.abs(cpsd)))

def tdoa(x, y, cc_func, fs=None):
    """Estimate time difference of arrival (TDOA) by finding peak in (G)CC.

    Args:
        x       : 1-d array, frequency domain signal 1
        y       : 1-d array, frequency domain signal 2
        cc_func : cross correlation function.
        fs      : sample rate, if not given (default) the result is number
                  of samples

    Returns:
        tdoa    : estimate of TDOA
    """
    cc = cc_func(x, y)
    peak_at = np.argmax(cc)
    if peak_at > len(cc) / 2:
        peak_at = peak_at - len(cc)
    if fs is None:
        return peak_at
    else:
        return 1.0 * peak_at / fs

def cc_across_time(tfx, tfy, cc_func):
    """Cross correlations across time.

    Args:
        tfx     : time-frequency domain signal 1.
        tfy     : time-frequency domain signal 2.
        cc_func : cross correlation function.

    Returns:
        cc_atime : cross correlation at different time.

    Note:
        If tfx and tfy are not of the same length, the result will be 
        truncated to the shorter one.
    """
    return np.array([cc_func(x, y) for x, y in izip(tfx, tfy)])

def pairwise_cc(tf, cc_func):
    """Pairwise cross correlations between all channels in signal.
    
    Args:
        tf      : multi-channel time-frequency domain signal.
        cc_func : cross correlation function.

    Returns:
        pw_cc   : pairwise cross correlations,
                  dict : (channel id, channel id) -> cross correlation across time.
    """
    nch = len(tf)
    return {(x, y) : cc_across_time(tf[x], tf[y], cc_func)
                for x in range(nch) for y in range(nch) if x < y}

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

