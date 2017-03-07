"""
cc.py

(general) cross correlations

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np

def gcc_phat(x, y):
    """GCC-PHAT

    Args:
        x  : 1-d array, frequency domain signal 1
        y  : 1-d array, frequency domain signal 2

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    r = x * y.conj()
    return np.real(np.fft.ifft(r / np.abs(r)))

def cross_correlation(x, y):
    """Cross correlation

    Args:
        x  : 1-d array, frequency domain signal 1
        y  : 1-d array, frequency domain signal 2

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    r = x * y.conj()
    return np.real(np.fft.ifft(r) / max(np.abs(r)))

def tdoa(x, y, cc_func, fs=None):
    """Estimate time difference of arrival (TDOA) by finding peak in (G)CC.

    Args:
        x       : 1-d array, frequency domain signal 1
        y       : 1-d array, frequency domain signal 2
        cc_func : cross correlation 
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

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

