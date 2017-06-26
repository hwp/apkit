"""
aspec_doa.py

Angular spectrum-based DOA estimation

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import scipy.ndimage

from .basic import steering_vector

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

    # apply to each channel pair
    ecov = np.zeros(cov.shape, dtype=cov.dtype)
    for i in xrange(len(tf)):
        for j in xrange(len(tf)):
            rpart = _apply_conv(cov[i,j,:,:].real, kernel, mode='nearest') 
            ipart = _apply_conv(cov[i,j,:,:].imag, kernel, mode='nearest')
            ecov[i,j,:,:] = rpart + 1j * ipart
    return ecov

def phi_mvdr(ecov, delay):
    """Local angular spectrum function: MVDR

    Args:
        ecov  : empirical covariance matrix, indices (cctf)
        delay : the set of delays to probe, indices (dc)

    Returns:
        phi   : local angular spectrum function, indices (dt),
                here 'd' is the index of delay
    """
    # compute inverse of empirical covariance matrix
    eta = 1e-10
    nch, _, nframe, nfbin = ecov.shape
    iecov = np.zeros(ecov.shape, dtype=ecov.dtype)
    for i in xrange(nframe):
        for j in xrange(nfbin):
            iecov[:,:,i,j] = np.asmatrix(ecov[:,:,i,j] + np.eye(nch) * eta).I

    phi = np.zeros((len(delay), nframe))
    for i in xrange(len(delay)):
        stv = steering_vector(delay[i], nfbin)
        denom = np.einsum('cf,cdtf,df->tf', stv.conj(), iecov, stv).real
        assert np.all(denom > 0)            # iecov positive definite
        phi[i] = np.sum(1. / denom, axis=1) # sum over frequency

    return phi

def local_maxima(phi, nlist, th=0.0):
    """Find local maxima

    Args:
        phi   : local angular spectrum function, indices (dt),
                here 'd' is the index of delay
        nlist : list of list of neighbor indices
        th    : the score of local maxima should exceed the threshold

    Returns:
        lmax  : list of local maxima indices, indexed by time.
    """
    ndoa, nframe = phi.shape
    lmax = []
    for pf in phi.T:
        lmf = []
        for i, p in enumerate(pf):
            if p > th:
                m = True
                for n in nlist[i]:
                    if p <= pf[n]:
                        m = False
                        break
                if m:
                    lmf.append(i)
        lmax.append(lmf)
    return lmax

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

