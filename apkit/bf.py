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

def compute_delay(m_pos, doa, c=340.29, fs=None):
    """Compute delay of signal arrival at microphones.

    Args:
        m_pos : microphone positions, (M,3) array,
                M is number of microphones.
        doa   : direction of arrival, (3,) array or (N,3) array,
                N is the number of sources.
        c     : (default 340.29 m/s) speed of sound.
        fs    : (default None) sample rate.

    Return:
        delay : delay with reference of arrival at first microphone.
                first element is always 0.
                unit is second if fs is not None, otherwise sample.
    """
    m_pos = np.asarray(m_pos)
    doa = np.asarray(doa)

    # relative position wrt first microphone
    r_pos = m_pos - m_pos[0]

    # inner product -> different in time 
    if doa.ndim == 1:
        diff = -np.einsum('ij,j->i', r_pos, doa) / c
    else:
        assert doa.ndim == 2
        diff = -np.einsum('ij,kj->ki', r_pos, doa) / c

    if fs is not None:
        return diff * fs
    else:
        return diff

def apply_beamforming(tf, bf_wt):
    """Apply beamforming to the signal with given beamforming weights
        
    Args:
        tf    : multi-channel time-frequency domain signal.
        bf_wt : beamforming weight

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    return np.einsum('ctf,cf->tf', tf, bf_wt)

def bf_weight_delay_sum(nfbin, delay, fs=None):
    """Compute weight of delay-sum beamformer

    Args:
        nfbin : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        fs    : sample rate. Default is None, @see delay.

    Returns:
        bf_wt : beamforming weight
    """
    nch = len(delay)

    # beamforming weight: delay and normalize
    return _steering_vec(delay, nfbin, fs).conj() / float(nch)

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
    _, _, nfbin = tf.shape

    # transfer function of delay filter
    bf_wt = bf_weight_delay_sum(nfbin, delay, fs)

    # apply transfer function and sum along channels
    return apply_beamforming(tf, bf_wt)

def bf_weight_superdir_fast(nfbin, delay, ninv, fs=None):
    """Compute weight of MVDR beamformer

    Args:
        nfbin : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        ninv : noise covariance inverse
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    # steering vector
    stv = _steering_vec(delay, nfbin, fs)

    # beamforming weight
    numerator = np.einsum('fcd,df->cf', ninv, stv)
    denominator = np.einsum('cf,cf->f', stv.conj(), numerator)
    return np.einsum('cf,f->cf', numerator, 1.0 / denominator).conj()

def bf_weight_superdir(nfbin, delay, ncov, fs=None):
    """Compute weight of MVDR beamformer

    Args:
        nfbin : number of frequency bins
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
    for i in xrange(nfbin):
        ninv[i] = np.asmatrix(ncov[i] + np.eye(nch) * eta).I

    return bf_weight_superdir_fast(nfbin, delay, ninv, fs)


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
    _, _, nfbin = tf.shape

    bf_wt = bf_weight_superdir(nfbin, delay, ncov, fs)

    # apply transfer function and sum along channels
    return apply_beamforming(tf, bf_wt)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

