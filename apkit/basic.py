"""
basic.py

basic functions

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import scipy.io.wavfile

def load_wav(filename):
    """Load wav file, convert to normalized float value

    Args:
        filename : string or open file handle.

    Returns:
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
    """
    fs, signal = scipy.io.wavfile.read(filename)
    if not np.issubdtype(signal.dtype, np.float):
        assert np.issubdtype(signal.dtype, np.integer)
        signal = signal.astype(float) / abs(np.iinfo(signal.dtype).min)
    return fs, signal.T

def save_wav(filename, fs, signal):
    """Save audio data as wave file.

    Args:
        filename : string or open file handle.
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
    """
    scipy.io.wavfile.write(filename, fs, signal.T)

def cola_hamming(win_size, hop_size):
    """ Hamming window, periodic and constant-overlap-add (COLA, sum=1)

    Args:
        win_size : window size
        hop_size : hop size

    Returns:
        w        : window coefficients
    """
    return np.hamming(win_size + 1)[0:win_size] \
                / 1.08 * hop_size / win_size * 2
    
def stft(signal, window, win_size, hop_size):
    """Convert time-domain signal to time-frequency domain.

    Args:
        signal   : multi-channel time-domain signal
        window   : window function, see cola_hamming as example.
        win_size : window size
        hop_size : hop size

    Returns:
        tf       : multi-channel time-frequency domain signal.
    """
    assert signal.ndim == 2
    w = window(win_size, hop_size)
    return np.array([[np.fft.fft(c[t:t+win_size] * w) 
        for t in xrange(0, len(c) - win_size, hop_size)] for c in signal])

def istft(tf, hop_size):
    """Inverse STFT

    Args:
        tf       : multi-channel time-frequency domain signal.
        window   : window function, see cola_hamming as example.
        win_size : window size
        hop_size : hop size

    Returns:
        signal   : multi-channel time-domain signal
    """
    tf = np.asarray(tf)
    nch, nframe, nfbin = tf.shape
    signal = np.zeros((nch, (nframe - 1) * hop_size + nfbin))
    for t in xrange(nframe):
        signal[:, t*hop_size:t*hop_size+nfbin] += \
                np.real(np.fft.ifft(tf[:, t]))
    return signal

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

