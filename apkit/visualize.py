"""
visualize.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_wave(fs, signal):
    """Convert time-domain signal to time-frequency domain.

    Args:
        fs     : sample rate.
        signal : multi-channel time-domain signal.
    """
    fig = plt.figure()

    for i, c in enumerate(signal):
        sp = fig.add_subplot(len(signal), 1, i+1)
        sp.plot(np.arange(len(c), dtype=float) / fs, c)

    plt.show()

def spectrogram(fs, tf, hop_size):
    """Convert time-domain signal to time-frequency domain.

    Args:
        fs    : sample rate.
        tf    : multi-channel time-frequency domain signal.
    """
    fig = plt.figure()

    vmax = np.max(np.abs(tf))
    for i, c in enumerate(tf):
        sp = fig.add_subplot(len(tf), 1, i+1)
        _, nfft = c.shape
        ny = nfft / 2 + 1
        im = sp.imshow(np.abs(c[:,:ny]).T, vmin=0, vmax=vmax,
                       interpolation='none', aspect='auto', origin='lower')
        '''cmap=plt.cm.YlOrRd,'''
        l = len(c) * hop_size / fs
        plt.xticks(np.arange(l + 1) * fs / hop_size,
                   [str(1.0 * t) for t in xrange(l + 1)])
        n = 4
        plt.yticks(np.arange(n + 1) * ny / n,
                   [str(fs / 2.0 * f / n) for f in np.arange(n + 1)])
        fig.colorbar(im)

    plt.show()

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

