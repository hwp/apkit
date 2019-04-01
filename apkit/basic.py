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
import scipy.ndimage

def load_wav(filename, offset=0, nsamples=-1):
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
    if offset > 0:
        w.setpos(offset)
    if nsamples < 0:
        nsamples = w.getnframes()
    data = np.fromstring(w.readframes(nsamples), dtype=dtype)
    data = data.reshape((-1, nchs))
    if not np.issubdtype(data.dtype, np.floating):
        assert np.issubdtype(data.dtype, np.integer)
        data = data.astype(float) / abs(np.iinfo(data.dtype).min)
    w.close()
    return fs, data.T

def load_metadata(filename):
    """Load metadata of a wav file instead of reading its content

    Args:
        filename : string or open file handle.

    Returns:
        fs        : sample rate.
        nchannels : number of channels
        nsamples  : number of samples
    """
    w = wave.open(filename, 'rb')
    nchs = w.getnchannels()
    fs = w.getframerate()
    nsamples = w.getnframes()
    w.close()
    return fs, nchs, nsamples

def save_wav(filename, fs, signal):
    """Save audio data as wav file.

    Args:
        filename : string or open file handle.
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
    """
    if np.issubdtype(signal.dtype, np.floating):
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

def stft(signal, window, win_size, hop_size, last_sample=False):
    """Convert time-domain signal to time-frequency domain.

    Args:
        signal   : multi-channel time-domain signal
        window   : window function, see cola_hamming as example.
        win_size : window size
        hop_size : hop size
        last_sample : include last sample, by default (due to legacy bug),
                      the last sample is not included.

    Returns:
        tf       : multi-channel time-frequency domain signal.
    """
    assert signal.ndim == 2
    w = window(win_size, hop_size)
    return np.array([[np.fft.fft(c[t:t+win_size] * w)
        for t in xrange(0, len(c) - win_size + (1 if last_sample else 0), hop_size)] for c in signal])

def istft(tf, hop_size):
    """Inverse STFT

    Args:
        tf       : multi-channel time-frequency domain signal.
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
    return np.einsum('ct,ct->c', signal, signal) / float(nsamples)

def power_avg(signal, vad_mask=None, vad_size=1):
    """Average (accros channels) power

    Args:
        signal   : multi-channel time-domain signal
        vad_mask : if given (default is None), the power on the voice
                   detected frames is computed.
        vad_size : vad frame size, default is 1.

    Returns:
        power    : average power of all channels.
    """
    return np.mean(power(signal, vad_mask, vad_size))

def power_db(signal, vad_mask=None, vad_size=1):
    """Power in dB

    Args:
        signal   : multi-channel time-domain signal
        vad_mask : if given (default is None), the power on the voice
                   detected frames is computed.
        vad_size : vad frame size, default is 1.

    Returns:
        power_db : power of each channel in dB.
    """
    return 10.0 * np.log10(power(signal, vad_mask, vad_size))

def power_avg_db(signal, vad_mask=None, vad_size=1):
    """Average (accros channels) power in dB

    Args:
        signal   : multi-channel time-domain signal
        vad_mask : if given (default is None), the power on the voice
                   detected frames is computed.
        vad_size : vad frame size, default is 1.

    Returns:
        power    : average power of all channels.
    """
    return 10.0 * np.log10(power_avg(signal, vad_mask, vad_size))

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

def steering_vector(delay, win_size=0, fbins=None, fs=None):
    """Compute the steering vector.

    One and only one of the conditions are true:
        - win_size != 0
        - fbins is not None

    Args:
        delay : delay of each channel (see compute_delay),
                unit is second if fs is not None, otherwise sample
        win_size : (default 0) window (FFT) size. If zero, use fbins.
        fbins : (default None) center of frequency bins, as discrete value.
        fs    : (default None) sample rate


    Returns:
        stv   : steering vector, indices (cf)
    """
    assert (win_size != 0) != (fbins is not None)
    delay = np.asarray(delay)
    if fs is not None:
        delay *= fs      # to discrete-time value
    if fbins is None:
        fbins = np.fft.fftfreq(win_size)
    return np.exp(-2j * math.pi * np.outer(delay, fbins))

def compute_delay(m_pos, doa, c=340, fs=None):
    """Compute delay of signal arrival at microphones.

    Args:
        m_pos : microphone positions, (M,3) array,
                M is number of microphones.
        doa   : normalized direction of arrival, (3,) array or (N,3) array,
                N is the number of sources.
        c     : (default 340) speed of sound (m/s).
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

def mel(f):
    """Mel function

    Args:
        f : frequency in Hz

    Returns:
        m : mel scale
    """
    return 1125.0 * np.log(1.0 + f / 700.0)

def mel_inv(m):
    """inverse mel function

    Args:
        m : mel scale

    Returns:
        f : frequency in Hz
    """
    return 700.0 * (np.exp(m / 1125.0) - 1.0)

def mel_freq_fbank_weight(n, freq, fs, fmax, fmin=0.0):
    """Mel-freqency filter banks weights

    Args:
        n    : number of filter banks
        freq : center of frequency bins as discrete value (-0.5 ~ 0.5),
               can be computed by numpy.fft.fftfreq
        fs   : sample rate
        fmax : maximal frequency in Hz
        fmin : (default 0) minimal frequency in Hz

    Returns:
        fbw  : filter bank weights, indexed by 'bf'.
               'b' is the index of filter bank.
    """
    mmax = mel(fmax)
    mmin = mel(fmin)
    mls = np.linspace(mmin, mmax, n + 2)
    fls = mel_inv(mls)
    fbw = np.zeros((n, len(freq)))
    freq = np.abs(fs * freq)

    # per bank
    for i in xrange(n):
        # left slope
        left = (freq - fls[i]) / (fls[i+1] - fls[i])
        left[left < 0.0] = 0.0
        left[left > 1.0] = 0.0
        # right slope
        right = (fls[i + 2] - freq) / (fls[i+2] - fls[i+1])
        right[right < 0.0] = 0.0
        right[right >= 1.0] = 0.0
        # sum
        fbw[i] = left + right

    assert np.min(fbw) == 0.0
    assert np.max(fbw) <= 1.0
    return fbw

def vad_by_threshold(fs, sig, vadrate, threshold_db, neighbor_size=0):
    """Voice Activity Detection by threshold

    Args:
        fs       : sample rate.
        signal   : multi-channel time-domain signal.
        vadrate  : output vad rate
        threshold_db : threshold in decibel
        neighbor_size : half size of (excluding center) neighbor area

    Returns:
        vad      : VAD label (0: silence, 1: active)
    """
    nch, nsamples = sig.shape
    nframes = nsamples * vadrate / fs
    fpower = np.zeros((nch, nframes)) # power at frame level
    for i in xrange(nframes):
        fpower[:, i] = power(sig[:, (i*fs/vadrate):((i+1)*fs/vadrate)])

    # average power in neighbor area
    if neighbor_size == 0:
        apower = fpower
    else:
        apower = np.zeros((nch, nframes))
        for i in xrange(nframes):
            apower[:, i] = np.mean(fpower[:, max(0,i-neighbor_size):
                                          min(nframes,i+neighbor_size+1)],
                                   axis=1)
    return (apower > 10.0 ** (threshold_db / 10.0)).astype(int)

def cov_matrix(tf):
    """Covariance matrix of the  multi-channel signal.

    Args:
        tf  : multi-channel time-frequency domain signal.

    Returns:
        cov : covariance matrix, indexed by (ccf)
    """
    nch, nframe, nfbin = tf.shape
    return np.einsum('itf,jtf->ijf', tf, tf.conj()) / float(nframe)

def empirical_cov_mat(tf, tw=2, fw=2):
    """Empirical covariance matrix

    Args:
        tf  : multi-channel time-frequency domain signal, indices (ctf)
        tw  : (default 2) half width of neighbor area in time domain,
              including center
        fw  : (default 2) half width of neighbor area in freq domain,
              including center

    Returns:
        ecov: empirical covariance matrix, indices (cctf)
    """
    _apply_conv = scipy.ndimage.filters.convolve

    # covariance matrix without windowing
    cov = np.einsum('ctf,dtf->cdtf', tf, tf.conj())

    # apply windowing by convolution
    # compute convolution window
    kernel = np.einsum('t,f->tf', np.hanning(tw * 2 + 1)[1:-1],
                       np.hanning(fw * 2 + 1)[1:-1])
    kernel = kernel / np.sum(kernel)    # normalize

    # apply to each channel pair
    ecov = np.zeros(cov.shape, dtype=cov.dtype)
    for i in xrange(len(tf)):
        for j in xrange(len(tf)):
            rpart = _apply_conv(cov[i,j,:,:].real, kernel, mode='nearest')
            ipart = _apply_conv(cov[i,j,:,:].imag, kernel, mode='nearest')
            ecov[i,j,:,:] = rpart + 1j * ipart
    return ecov

def empirical_cov_mat_by_block(tf, block_size, block_hop):
    """Empirical covariance matrix by blocks

    Args:
        tf  : multi-channel time-frequency domain signal, indices (ctf)
        block_size : number of frames in one block
        block_hop  : number of frame shifts between blocks

    Returns:
        ecov: empirical covariance matrix, indices (cctf)
    """
    nch, nframe, nfbin = tf.shape

    # covariance matrix
    cov = np.einsum('ctf,dtf->cdtf', tf, tf.conj())

    # average in blocks
    ecov = [np.mean(cov[:,:,t:t+block_size], axis=2)
                for t in xrange(0, nframe - block_size + 1, block_hop)]
    ecov = np.moveaxis(np.asarray(ecov), 0, 2)
    return ecov

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

