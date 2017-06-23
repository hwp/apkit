"""
aspec_doa.py

Angular spectrum-based DOA estimation

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import scipy.ndimage

_apply_conv = scipy.ndimage.filters.convolve

def empirical_cov_mat(tf, tw=2, fw=2):
    """Empirical covariance matrix

    Args:
        tf  : multi-channel time-frequency domain signal, indices (ctf)
        tw  : (default 1) half width of neighbor area in time domain,
              excluding center
        fw  : (default 1) half width of neighbor area in freq domain,
              excluding center

    Returns:
        ecov: empirical covariance matrix, indices (cctf)
    """
    # covariance matrix without windowing
    cov = np.einsum('ctf,dtf->cdtf', tf, tf.conj())

    # apply windowing by convolution
    # compute convolution window
    kernel = np.einsum('t,f->tf', np.hanning(tw * 2 + 1)[1:-1],
                       np.hanning(fw * 2 + 1)[1:-1])
    kernel = kernel / np.sum(kernel)    # normalize
    print kernel

    # apply to each channel pair
    ecov = np.zeros(cov.shape, dtype=cov.dtype)
    for i in xrange(len(tf)):
        for j in xrange(len(tf)):
            rpart = _apply_conv(cov[i,j,:,:].real, kernel, mode='nearest') 
            ipart = _apply_conv(cov[i,j,:,:].imag, kernel, mode='nearest')
            ecov[i,j,:,:] = rpart + 1j * ipart
    return ecov

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

