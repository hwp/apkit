"""
bf.py

Beamforming

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import math

import numpy as np

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
    delay = np.asarray(delay)
    print delay

    if fs is not None:
        delay *= fs      # to discrete-time value
    nch, _, nfbin = tf.shape
    print delay
    print tf.shape
    freq = np.fft.ifftshift(
            (np.arange(nfbin, dtype=float) - nfbin / 2) / nfbin)
    print freq.shape
    # transfer function of delay
    tr_func = np.exp(2j * math.pi * np.outer(delay, freq)) / nch
    
    print tr_func
    tcf = np.swapaxes(tf, 0, 1)     # indices: time, channel, frequency

    print tcf.shape
    # apply transfer function and sum along channels
    return np.sum(tcf * tr_func, axis=1)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

