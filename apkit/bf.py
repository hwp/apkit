"""
bf.py

Beamforming

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np

from .basic import steering_vector

def apply_beamforming(tf, bf_wt):
    """Apply beamforming to the signal with given beamforming weights
        
    Args:
        tf    : multi-channel time-frequency domain signal.
        bf_wt : beamforming weight

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    return np.einsum('ctf,cf->tf', tf, bf_wt.conj())

def bf_weight_delay_sum(win_size, delay, fs=None):
    """Compute weight of delay-sum beamformer

    Args:
        win_size : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        fs    : sample rate. Default is None, @see delay.

    Returns:
        bf_wt : beamforming weight, indexed by (cf)
    """
    nch = len(delay)

    # beamforming weight: delay and normalize
    return steering_vector(delay, win_size, fs=fs) / float(nch)

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
    _, _, win_size = tf.shape

    # transfer function of delay filter
    bf_wt = bf_weight_delay_sum(win_size, delay, fs)

    # apply transfer function and sum along channels
    return apply_beamforming(tf, bf_wt)

def bf_weight_superdir_fast(win_size, delay, ninv, fs=None):
    """Compute weight of MVDR beamformer

    Args:
        win_size : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        ninv : noise covariance inverse
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    # steering vector
    stv = steering_vector(delay, win_size, fs=fs)

    # beamforming weight
    numerator = np.einsum('fcd,df->cf', ninv, stv)
    denominator = np.einsum('cf,cf->f', stv.conj(), numerator)
    return np.einsum('cf,f->cf', numerator, 1.0 / denominator)

def bf_weight_superdir(win_size, delay, ncov, fs=None):
    """Compute weight of MVDR beamformer

    Args:
        win_size : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        ncov  : noise covariance
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    nch = len(delay)
    eta = 1e-6

    ninv = np.zeros(ncov.shape, dtype=complex) 
    for i in xrange(win_size):
        ninv[i] = np.asmatrix(ncov[i] + np.eye(nch) * eta).I

    return bf_weight_superdir_fast(win_size, delay, ninv, fs)


def bf_superdir(tf, delay, ncov, fs=None):
    """Apply MVDR beamformer to signals.

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
    _, _, win_size = tf.shape

    bf_wt = bf_weight_superdir(win_size, delay, ncov, fs)

    # apply transfer function and sum along channels
    return apply_beamforming(tf, bf_wt)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

