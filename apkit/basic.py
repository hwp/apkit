"""
basic.py

basic functions

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import os
import math
import wave

import numpy as np

def load_wav(filename):
    """Load wav file, convert to normalized float value

    Args:
        filename : string or open file handle.

    Returns:
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
    """
    w = wave.open(filename, 'rb')
    nchs = w.getnchannels()
    fs = w.getframerate()
    sw = w.getsampwidth()
    if sw == 2:
        dtype = np.int16
    elif sw == 1:
        dtype = np.int8
    elif sw == 4:
        dtype = np.int32
    else:
        assert False
    data = np.fromstring(w.readframes(w.getnframes()), dtype=dtype)
    data = data.reshape((-1, nchs))
    if not np.issubdtype(data.dtype, np.float):
        assert np.issubdtype(data.dtype, np.integer)
        data = data.astype(float) / abs(np.iinfo(data.dtype).min)
    w.close()
    return fs, data.T

def save_wav(filename, fs, signal):
    """Save audio data as wav file.

    Args:
        filename : string or open file handle.
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
    """
    if np.issubdtype(signal.dtype, float):
        signal[signal > 1.0] = 1.0
        signal[signal < -1.0] = -1.0
        dtype = np.dtype('int16')
        signal = (signal * np.iinfo(dtype).max).astype(dtype)

    w = wave.open(filename, 'wb')
    w.setnchannels(len(signal))
    w.setsampwidth(signal.dtype.itemsize)
    w.setframerate(fs)
    w.writeframes(signal.T.tobytes())
    w.close()

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

def cola_rectangle(win_size, hop_size):
    """ Recangle window, periodic and constant-overlap-add (COLA, sum=1)

    Args:
        win_size : window size
        hop_size : hop size

    Returns:
        w        : window coefficients
    """
    return np.ones(win_size) * hop_size / win_size

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

def freq_upsample(s, upsample):
    """ padding in frequency domain, should be used with ifft so that
    signal is upsampled in time-domain.

    Args:
        s        : frequency domain signal
        upsample : an integer indicating factor of upsampling.

    Returns:
        padded signal
    """
    if upsample == 1:
        return s
    assert isinstance(upsample, int) and upsample > 1
    l = len(s)
    if l % 2 == 0:
        h = l / 2
        return upsample * np.concatenate(
                (s[:h], np.array([s[h] / 2.0]),
                 np.zeros(l * (upsample - 1) - 1),
                 np.array([s[h] / 2.0]), s[h+1:]))
    else:
        h = l / 2 + 1
        return upsample * np.concatenate(
                (s[:h], np.zeros(l * (upsample - 1)), s[h:]))

def power(signal, vad_mask=None, vad_size=1):
    """Signal power

    Args:
        signal   : multi-channel time-domain signal
        vad_mask : if given (default is None), the power on the voice
                   detected frames is computed.
        vad_size : vad frame size, default is 1.

    Returns:
        power    : power of each channel.
    """
    nch, nsamples = signal.shape
    if vad_mask is not None:
        vad_mask = vad_mask.repeat(vad_size)
        if len(vad_mask) >= nsamples:
            vad_mask = vad_mask[:nsamples]
        else:
            vad_mask = np.append(vad_mask,
                                 np.zeros(nsamples - len(vad_mask))
                                    .astype(np.bool))
        signal = signal[:,vad_mask]
    return np.einsum('ct,ct->c', signal, signal) / float(len(signal.T))

def power_tf(tf):
    """Compute power of time-frequency domain signal

    Args:
        tf       : mono/multi-channel time-frequency domain signal.

    Returns:
        power    : power of each channel.
    """
    if tf.ndim == 2:
        # mono channel
        nt, nf = tf.shape
        return np.einsum('tf,tf', tf, tf.conj()).real / float(nt * nf)
    else:
        # multi channel
        nch, nt, nf = tf.shape
        return np.einsum('ctf,ctf->c', tf, tf.conj()).real / float(nt * nf)

def snr(sandn, noise):
    """Signal-to-noise ratio given signal with noise and noise

    Args:
        sandn : signal and noise multi-channel time-domain signal
        noise : noise multi-channel time-domain signal

    Returns:
        snr   : snr of each channel in dB.
    """
    pnos = power(noise)
    psig = power(sandn) - pnos
    return 10 * np.log10(psig / pnos)

def steering_vector(delay, nfbin, fs=None):
    """Steering vector of delay.

    Args:
        delay : delay of each channel,
                unit is second if fs is not None, otherwise sample
        nfbin : number of frequency bins, aka window size
        fs    : (default None) sample rate

    Returns:
        stv   : steering vector, indices (cf)
    """
    delay = np.asarray(delay)
    if fs is not None:
        delay *= fs      # to discrete-time value
    freq = np.fft.fftfreq(nfbin)
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
                unit is second if fs is None, otherwise sample.
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

def load_pts_on_sphere(name='p4000'):
    """Load points on a unit sphere

    Args:
        name : should always be 'p4000'

    Returns:
        pts  : array of points on a unit sphere
    """
    this_dir, this_filename = os.path.split(__file__)
    data_path = os.path.join(this_dir, 'data', '%s.npy' % name)
    return np.load(data_path)

def neighbor_list(pts, dist, scale_z=1.0):
    """List of neighbors

    Args:
        pts     : array of points on a unit sphere
        dist    : distance (rad) threshold
        scale_z : (default 1.0) scale of z-axis,
                  if scale_z is smaller than 1, more neighbors will be 
                  along elevation

    Returns:
        nlist   : list of list of neighbor indices
    """
    # pairwise inner product
    if scale_z != 1.0:
        pts = np.copy(pts)
        pts[:,2] *= scale_z
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pip = np.einsum('ik,jk->ij', pts, pts)

    # adjacency matrix
    amat = pip >= math.cos(dist)
    for i in xrange(len(pts)):
        amat[i,i] = False

    # convert to list
    return [list(np.nonzero(n)[0]) for n in amat]

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

