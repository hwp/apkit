"""
cc.py

(general) cross correlations

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
from itertools import izip

def _freq_upsample(s, upsample):
    """ padding in frequency domain, should be used with ifft so that
    signal is upsampled in time-domain.

    Args:
        s        : frequency domain signal
        upsample : an integer indicating factor of upsampling.

    Returns:
        padded signal
    """
    if upsample == 1:
        return s
    assert isinstance(upsample, int) and upsample > 1
    l = len(s)
    if l % 2 == 0:
        h = l / 2
        return upsample * np.concatenate(
                (s[:h], np.array([s[h] / 2.0]),
                 np.zeros(l * (upsample - 1) - 1),
                 np.array([s[h] / 2.0]), s[h+1:]))
    else:
        h = l / 2 + 1
        return upsample * np.concatenate(
                (s[:h], np.zeros(l * (upsample - 1)), s[h:]))

def gcc_phat(x, y, upsample=1):
    """GCC-PHAT

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        upsample : an integer indicating factor of upsampling.
                   default value is 1.

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    cpsd = x * y.conj()
    cpsd_phat = cpsd / np.abs(cpsd)
    cpsd_phat = _freq_upsample(cpsd_phat, upsample)
    return np.real(np.fft.ifft(cpsd_phat))

def cross_correlation(x, y, upsample=1):
    """Cross correlation (vanilla)

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        upsample : an integer indicating factor of upsampling.
                   default value is 1.

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    cpsd = _freq_upsample(x * y.conj(), upsample)
    return np.real(np.fft.ifft(cpsd) / np.max(np.abs(cpsd)))

def cc_across_time(tfx, tfy, cc_func, cc_args=()):
    """Cross correlations across time.

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        cc_func  : cross correlation function.
        cc_args : extra arguments of cc_func.

    Returns:
        cc_atime : cross correlation at different time.

    Note:
        If tfx and tfy are not of the same length, the result will be 
        truncated to the shorter one.
    """
    return np.array([cc_func(x, y, *cc_args) for x, y in izip(tfx, tfy)])

def pairwise_cc(tf, cc_func, cc_args=()):
    """Pairwise cross correlations between all channels in signal.
    
    Args:
        tf      : multi-channel time-frequency domain signal.
        cc_func : cross correlation function.
        cc_args : extra arguments of cc_func.

    Returns:
        pw_cc   : pairwise cross correlations,
                  dict : (channel id, channel id) -> cross correlation across time.
    """
    nch = len(tf)
    return {(x, y) : cc_across_time(tf[x], tf[y], cc_func, cc_args)
                for x in range(nch) for y in range(nch) if x < y}

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

