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
                       cmap=plt.get_cmap('magma'), interpolation='none',
                       aspect='auto', origin='lower')
        l = len(c) * hop_size / fs
        plt.xticks(np.arange(l + 1) * fs / hop_size,
                   [str(1.0 * t) for t in xrange(l + 1)])
        n = 4
        plt.yticks(np.arange(n + 1) * ny / n,
                   [str(fs / 2.0 * f / n) for f in np.arange(n + 1)])
        fig.colorbar(im)

def plot_cc(fs, pw_cc, hop_size, ch_names=None, zoom=None, upsample=1):
    """Plot pairwise cross correlation across time.

    Args:
        fs       : sample rate of the original signal (before upsampling).
        pw_cc    : pairwise cross correlation, result from apkit.pairwise_cc
        hop_size : hop size of the STFT.
        zoom     : if not None, zoom along y-axis (time-difference) 
                   to +/- zoom samples.
        upsample : factor of upsampling used for computing the cross
                   correlation.
    """
    fig = plt.figure()

    keys = sorted(pw_cc.keys())
    for i, k in enumerate(keys):
        cc = pw_cc[k]
        sp = fig.add_subplot(len(pw_cc), 1, i+1)
        _, nfft = cc.shape
        ny = nfft / 2 + 1
        assert zoom < ny
        vmax = np.max(np.abs(cc))

        if zoom is None:
            cc = np.concatenate((cc[:,ny:], cc[:,:ny]), axis=1)
        else:
            cc = np.concatenate((cc[:,-zoom:],
                                 cc[:,:zoom+1]), axis=1)
            ny = zoom
        im = sp.imshow(cc.T, vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr,
                       interpolation='none', aspect='auto',
                       origin='lower')
        if ch_names is None:
            plt.title('Channel %d vs. %d' % k)
        else:
            plt.title('%s vs. %s' % tuple(ch_names[c] for c in k))

        l = len(cc) * hop_size / fs
        m = 1
        while l > 10:
            l = l / 10
            m = m * 10
        plt.xticks(np.arange(l + 1) * m * fs / hop_size,
                   [str(1.0 * m * t) for t in xrange(l + 1)])

        n = 2
        ypos = ny + np.arange(-n,n+1) * ny / n
        plt.yticks(ypos,
                   ['%.2g' % y for y in (ypos - ny) * 1.0 / (upsample * fs)])
        fig.colorbar(im)

def show():
    plt.show()

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

