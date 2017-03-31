"""
bf.py

Beamforming

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import math

import numpy as np

def _steering_vec(delay, nfbin, fs=None):
    """Steering vector of delay.
    """
    delay = np.asarray(delay)
    if fs is not None:
        delay *= fs      # to discrete-time value
    freq = np.fft.ifftshift(
            (np.arange(nfbin, dtype=float) - nfbin / 2) / nfbin)
    return np.exp(-2j * math.pi * np.outer(delay, freq))

def bf_delay_sum(tf, delay, fs=None):
    """Apply delay-sum beamformer to signals.

    Args:
        tf    : multi-channel time-frequency domain signal.
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    tf = np.asarray(tf)
    nch, _, nfbin = tf.shape

    # transfer function of delay filter
    tr_func = _steering_vec(delay, nfbin, fs).conj() / float(nch)

    # apply transfer function and sum along channels
    return np.einsum('ctf,cf->tf', tf, tr_func)

def bf_superdir(tf, delay, ncov, fs=None):
    """Apply superdirective beamformer to signals.

    Args:
        tf    : multi-channel time-frequency domain signal.
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        ncov  : noise covariance
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    tf = np.asarray(tf)
    nch, _, nfbin = tf.shape

    eta = 1e-6

    ninv = np.zeros(ncov.shape, dtype=complex) 
    for i in xrange(nfbin):
        ninv[i] = np.asmatrix(ncov[i] + np.eye(nch) * eta).I

    stv = _steering_vec(delay, nfbin, fs)

    # transfer function
    numerator = np.einsum('fcd,df->cf', ninv, stv)
    denominator = np.einsum('cf,cf->f', stv.conj(), numerator)
    tr_func = np.einsum('cf,f->cf', numerator, 1.0 / denominator).conj()
    
    # apply transfer function and sum along channels
    return np.einsum('ctf,cf->tf', tf, tr_func)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

