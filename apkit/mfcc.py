"""
mfcc.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import scipy

from .basic import mel_freq_fbank_weight


def mfcc(tf, n_mfcc, fs, fmin=0.0, fmax=None):
    """
    Extract MFCC vectors

    Args:
        tf   : single-channel time-frequency domain signal,
               indexed by 'tf'
        n_mfcc : number of coefficients
        fs   : sample rate
        fmin : (default 0) minimal frequency in Hz
        fmax : (default fs/2) maximal frequency in Hz

    Returns:
        mfcc : MFCC
    """
    if fmax is None:
        fmax = fs / 2.0
    n_frame, n_fbin = tf.shape

    # get filter weights
    freq = np.fft.fftfreq(n_fbin)
    fbw = mel_freq_fbank_weight(n_mfcc, freq, fs, fmin=fmin, fmax=fmax)

    # get log power
    sigpow = np.real(tf * tf.conj())
    logfpow = np.log(np.einsum('bf,tf->tb', fbw, sigpow) + 1e-20)

    # DCT
    mfcc = scipy.fft.dct(logfpow)
    return mfcc
