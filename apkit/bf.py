"""
bf.py

Beamforming

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import abc

import numpy as np
import scipy.linalg

from .basic import empirical_cov_mat

def apply_beamforming(tf, bf_wt):
    """Apply beamforming to the signal with given beamforming weights

    Args:
        tf    : multi-channel time-frequency domain signal.
        bf_wt : beamforming weight

    Returns:
        res   : filtered signal in time-frequency domain.
    """
    return np.einsum('ctf,cf->tf', tf, bf_wt.conj())

class AbstractBeamformer(object):
    __metaclass__  = abc.ABCMeta

    def load_bg_cov(self, cov):
        """Load background signal covariance matrix

        Args:
            cov : covariance matrix (indexed by ccf)
        """
        pass

    def preload_tf(self, tf):
        """Pre-load signal for faster processing
            (in case of beamform on multiple directions of the same signal)

        Args:
            tf  : multi-channel time-frequency domain signal.
        """
        pass

    @abc.abstractmethod
    def __call__(self, tf, stv):
        """Apply beamformer

        Args:
            tf  : multi-channel time-frequency domain signal.
            stv : steering vector computed by steering_vector

        Returns:
            res : filtered signal in one-channel time-frequency domain,
                  (in multi-channel format -- indexed by ctf).
        """
        pass

class DelaySum(AbstractBeamformer):
    """delay-sum beamformer"""

    def __call__(self, tf, stv):
        bf_wt = stv / float(len(stv))
        return np.expand_dims(apply_beamforming(tf, bf_wt), axis=0)

class StaticMVDR(AbstractBeamformer):
    """static MVDR beamformer

    static (c.f. adaptive) means the covariance matrix is fixed and pre-computed
    with background signal
    """
    def load_bg_cov(self, cov):
        nch, _, nfbins = cov.shape
        eta = 1e-6

        self.binv = np.zeros(cov.shape, dtype=complex)
        for i in xrange(nfbins):
            self.binv[:,:,i] = np.asmatrix(cov[:,:,i] + np.eye(nch) * eta).I

    def __call__(self, tf, stv):
        numerator = np.einsum('cdf,df->cf', self.binv, stv)
        denominator = np.einsum('cf,cf->f', stv.conj(), numerator)
        bf_wt = np.einsum('cf,f->cf', numerator, 1.0 / denominator)

        # apply transfer function and sum along channels
        return np.expand_dims(apply_beamforming(tf, bf_wt), axis=0)

class MVDR(AbstractBeamformer):
    """(adaptive) MVDR beamformer

    covariance matrix are computed for each time-frequency neighborhood
    """
    def __init__(self, eta=1e-6):
        """
        Args:
            eta : a constant added to the diagonal of the covariance matrix,
                  it is the so-called white noise constraint
        """
        self.eta = eta

    def preload_tf(self, tf):
        nch, _, _ = tf.shape
        wnc = (np.eye(nch) * self.eta).reshape(nch, nch, 1, 1)
                                            # white noise constraint
        self.ecov = empirical_cov_mat(tf, tw=2, fw=2) + wnc

    def __call__(self, tf, stv):
        nch, nframe, nfbin = tf.shape
        res = np.zeros((1, nframe, nfbin), dtype=tf.dtype)

        for i in xrange(nframe):
            numerator = np.zeros(stv.shape, stv.dtype)
            for j in xrange(nfbin):
                numerator[:,j] = scipy.linalg.solve(self.ecov[:,:,i,j], stv[:,j],
                                                    assume_a='pos')
            denominator = np.einsum('cf,cf->f', stv.conj(), numerator)
            bf_wt = np.einsum('cf,f->cf', numerator, 1.0 / denominator)

            # apply for each frame
            res[0,i] = apply_beamforming(tf[:,[i]], bf_wt)
        return res


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

