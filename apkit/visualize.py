"""
visualize.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_wave(fs, signal):
    """Plot wave of singals.

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
    """Plot spectrograms of singals

    Args:
        fs       : sample rate.
        tf       : multi-channel time-frequency domain signal.
        hop_size : hop size of the STFT.
    """
    fig = plt.figure()

    vmax = np.percentile(np.abs(tf), 99)
    for i, c in enumerate(tf):
        sp = fig.add_subplot(len(tf), 1, i+1)
        _, nfft = c.shape
        ny = nfft / 2 + 1
        im = sp.imshow(np.abs(c[:,:ny]).T, vmin=0, vmax=vmax,
                       cmap=plt.cm.YlOrRd, interpolation='none',
                       aspect='auto', origin='lower')
        l = len(c) * hop_size / fs
        plt.xticks(np.arange(l + 1) * fs / hop_size,
                   [str(1.0 * t) for t in xrange(l + 1)])
        n = 4
        plt.yticks(np.arange(n + 1) * ny / n,
                   [str(fs / 2.0 * f / n) for f in np.arange(n + 1)])
        fig.colorbar(im)

    plt.show()

def cc_graph(fs, pw_cc, hop_size):
    """Plot pairwise cross correlation across time.

    Args:
        fs       : sample rate.
        pw_cc    : pairwise cross correlation, result from apkit.pairwise_cc
        hop_size : hop size of the STFT.
    """
    fig = plt.figure()

    for i, (k, cc) in enumerate(pw_cc.items()):
        sp = fig.add_subplot(len(pw_cc), 1, i+1)
        _, nfft = cc.shape
        ny = nfft / 2 + 1
        vmax = np.max(np.abs(cc))
        im = sp.imshow(np.concatenate((cc[:,ny:], cc[:,:ny]), axis=1).T,
                       vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr,
                       interpolation='none', aspect='auto',
                       origin='lower')
        l = len(cc) * hop_size / fs
        plt.xticks(np.arange(l + 1) * fs / hop_size,
                   [str(1.0 * t) for t in xrange(l + 1)])
        '''
        n = 4
        plt.yticks(np.arange(n + 1) * ny / n,
                   [str(fs / 2.0 * f / n) for f in np.arange(n + 1)])
        '''
        fig.colorbar(im)

    plt.show()


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

