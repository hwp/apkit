"""
cc.py

(general) cross correlations

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import math
from itertools import izip

import numpy as np

from .basic import freq_upsample

def gcc_phat(x, y, upsample=1, noise_cpsd=None, eps=0.0):
    """GCC-PHAT

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        upsample : an integer indicating factor of upsampling.
                   default value is 1.
        noise_cpsd : noise cross power spectral density
        eps      : (default 0.0) small constant added to denominator for
                   numerical stability, as well as to suppress low engergy
                   bins.

    Returns:
        cc : cross correlation of the two signal, 1-d array,
             index corresponds to time-domain signal
    """
    cpsd = x.conj() * y
    if noise_cpsd is not None:
        cpsd = cpsd - noise_cpsd
    # phat transform
    if eps <= 0.0:
        cpsd_phat = cpsd * 1.0      # copy
        cpsd_phat[cpsd != 0] /= np.abs(cpsd[cpsd != 0])
    else:
        cpsd_phat = cpsd / (np.abs(cpsd) + eps)
    cpsd_phat = freq_upsample(cpsd_phat, upsample)
    return np.real(np.fft.ifft(cpsd_phat))

def gcc_phat_fbanks(ecov, fbw, zoom, freq=None, eps=0.0):
    """GCC-PHAT on filter banks

    Args:
        ecov : empirical covariance matrix, computed by empirical_cov_mat
        fbw  : filter bank weights, indexed by bf
               possibly computed by mel_freq_fbank_weight
        zoom : number of center coefficients on each side, excluding center
        freq : (default None) center of frequency bins as discrete
               value (-0.5 ~ 0.5). By default it is computed by 
               numpy.fft.fftfreq with fft size
        eps  : (default 0.0) small constant added to denominator for
               numerical stability, as well as to suppress low engergy
               bins.
    Return:
        fbcc : pairwise GCC-PHAT on filter banks,
               dictionary with keys same as pairwise_cc.
               each element is indexed by 'btd'
    """
    nch, _, nframe, nfbin = ecov.shape
    assert fbw.shape[1] == nfbin

    if freq is None:
        freq = np.fft.fftfreq(nfbin)
    delay = np.arange(-zoom, zoom + 1, dtype=float)
    steer = np.exp(-2j * math.pi * np.outer(freq, delay))

    # normalize weight
    fbwn = fbw / np.sum(fbw, axis=1, keepdims=True) 
    fbcc = {}
    for i in xrange(nch - 1):
        for j in xrange(i + 1, nch):
            # phase transform on CPSD
            cpsd = ecov[i, j]
            if eps <= 0.0:
                cpsd_phat = cpsd * 1.0      # copy
                cpsd_phat[cpsd != 0] /= np.abs(cpsd[cpsd != 0])
            else:
                cpsd_phat = cpsd / (np.abs(cpsd) + eps)

            # weighted sum
            '''
            cc = np.einsum('bf,tf,fd->btd', fbwn, cpsd_phat, steer,
                           optimize='optimal').real
            '''
            cc = np.zeros((len(fbw), nframe, len(delay)))
            for b in xrange(len(fbw)):
                for f in xrange(nfbin):
                    if fbwn[b,f] > 0:
                        cc[b,:,:] += fbwn[b,f] * \
                                        np.outer(cpsd_phat[:,f], steer[f,:]).real
            fbcc[(i, j)] = cc

    return fbcc

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
    cpsd = freq_upsample(x.conj() * y, upsample)
    return np.real(np.fft.ifft(cpsd))

def cc_across_time(tfx, tfy, cc_func, cc_args=()):
    """Cross correlations across time.

    Args:
        x        : 1-d array, frequency domain signal 1
        y        : 1-d array, frequency domain signal 2
        cc_func  : cross correlation function.
        cc_args  : list of extra arguments of cc_func.

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
                  if cc_args is a dict, the keys will be indices of 
                  channel pairs, and the value will be pass to the cc_func
                  for this pair. Otherwise, cc_args will be regarded as a
                  list and same arguments will be used for all pairs.

    Returns:
        pw_cc   : pairwise cross correlations,
                  dict : (channel id, channel id) -> cross correlation across time.
    """
    nch = len(tf)
    return {(x, y) : cc_across_time(tf[x], tf[y], cc_func, 
                cc_args[(x, y)] if isinstance(cc_args, dict) else cc_args)
                for x in range(nch) for y in range(nch) if x < y}

def pairwise_cpsd(tf):
    """Pairwise cross power spectral density between all channels in singal.
    The result is averaged across time.

    Args:
        tf      : multi-channel time-frequency domain signal.

    Returns:
        pw_cpsd : pairwise cross power spectral density.
                  dict : (channel id, channel id) -> cpsd
    """
    nch = len(tf)
    return {(x, y) : np.average(tf[x].conj() * tf[y], axis=0)
                for x in range(nch) for y in range(nch) if x < y}

def cov_matrix(tf):
    """Covariance matrix of the  multi-channel signal.

    Args:
        tf  : multi-channel time-frequency domain signal.

    Returns:
        cov : covariance matrix, indexed by (ccf)
    """
    nch, nframe, nfbin = tf.shape
    return np.einsum('itf,jtf->ijf', tf, tf.conj()) / float(nframe)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

