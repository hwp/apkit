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

class _BfWeightDelaySum:
    def need_cov(self):
        return False

    def __call__(self, stv, cov=None):
        """Compute weight of delay-sum beamformer

        Args:
            stv  : steering vector, indexed by (cf)
            cov  : covariance matrix, not used by this function

        Returns:
            bf_wt : beamforming weight, indexed by (cf)
        """
        return stv / float(len(stv))

bf_weight_delay_sum = _BfWeightDelaySum()

class _BfWeightMvdr:
    def need_cov(self):
        return True

    def __call__(self, stv, cov):
        """Compute weight of delay-sum beamformer

        Args:
            stv  : steering vector, indexed by (cf)
            cov  : covariance matrix, indexed by (ccf)

        Returns:
            bf_wt : beamforming weight, indexed by (cf)
        """
        # move frequency axis to first and diagonal correction
        cov_fcc = np.moveaxis(cov, -1, 0) + np.eye(len(stv)) * 1e-15

        # R^-1 a
        rinva = np.linalg.solve(cov_fcc, np.moveaxis(stv, -1, 0))

        # a^H R^-1 a
        ahrinva = np.einsum('cf,fc->f', stv.conj(), rinva)

        # 1 / (a^H R^-1 * a) * (R^-1 a)
        return np.einsum('f,fc->cf', 1.0 / ahrinva, rinva)

bf_weight_mvdr = _BfWeightMvdr()

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
    nch, _, win_size = tf.shape

    # steering vector
    stv = steering_vector(delay, win_size, fs=fs)

    # transfer function of delay filter
    bf_wt = bf_weight_delay_sum(stv)

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

def bf_weight_superdir(win_size, delay, cov, fs=None):
    """Compute weight of MVDR beamformer

    Args:
        win_size : number of frequency bins
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        cov   : covariance matrix, indexed by (ccf)
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    nch = len(delay)
    eta = 1e-6

    ninv = np.zeros(cov.shape, dtype=complex) 
    for i in xrange(win_size):
        ninv[i] = np.asmatrix(cov[i] + np.eye(nch) * eta).I

    return bf_weight_superdir_fast(win_size, delay, ninv, fs)


def bf_superdir(tf, delay, cov, fs=None):
    """Apply (static) MVDR beamformer to signals.

    Args:
        tf    : multi-channel time-frequency domain signal.
        delay : delay of each channel. If fs is given, value denotes
                delay in second (continuous-time) ,
                otherwise # of samples (discrete-time).
        cov   : covariance matrix, indexed by (ccf)
        fs    : sample rate. Default is None, @see delay.

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    tf = np.asarray(tf)
    _, _, win_size = tf.shape

    bf_wt = bf_weight_superdir(win_size, delay, cov, fs)

    # apply transfer function and sum along channels
    return apply_beamforming(tf, bf_wt)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

