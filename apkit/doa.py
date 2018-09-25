"""
doa.py

Direction of arrival (DOA) estimation.

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import scipy

_TOLERANCE = 1e-13

def vec2ae(v):
    """Compute the azimuth and elevation of vectors

    Args:
        v : vector or list of vectors

    Returns:
        if one vector given:
            azimuth   : angle in radian in the x-y plane
            elevation : angle in radian from x-y plane to vector
        else (list of vectors):
            list of (azimuth, elevation)
    """
    v = np.asarray(v)
    if v.ndim == 1:
        x, y, z = v
    else:
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]
    n = np.sqrt(x ** 2 + y ** 2)
    return np.asarray([np.arctan2(y, x), np.arctan2(z, n)]).T

def vec2xsyma(v):
    """Compute the angle between the vector and the x-axis

    Args:
        v : vector or list of vectors

    Returns:
        if one vector given:
            angle (in radian)
        else (list of vectors):
            list of angles
    """
    v = np.asarray(v)
    if v.ndim == 1:
        x, y, z = v
    else:
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]
    n = np.sqrt(y ** 2 + z ** 2)
    return np.arctan2(n, x)

def vec2ysyma(v):
    """Compute the angle between the vector and the y-axis - pi/2

    Args:
        v : vector or list of vectors

    Returns:
        if one vector given:
            angle (in radian)
        else (list of vectors):
            list of angles
    """
    v = np.asarray(v)
    if v.ndim == 1:
        x, y, z = v
    else:
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]
    n = np.sqrt(x ** 2 + z ** 2)
    return np.arctan2(y, n)

def _u_sqr_minus_1(lam, m, tau):
    n, d = m.shape
    try:
        u = np.linalg.solve(m.T * m - lam * np.eye(d), m.T * tau)
    except np.linalg.LinAlgError as e:
        print e
        # TODO handle singular case
        return -1
    return np.asscalar(u.T * u - 1)

def _error_func(u, m, tau):
    if not isinstance(u, np.matrix):
        u = np.asmatrix(u).T
    d = tau - m * u
    return np.asscalar(d.T * d)

def _constraint_func(u):
    if not isinstance(u, np.matrix):
        u = np.asmatrix(u).T
    return 1.0 - np.asscalar(u.T * u)

def doa_least_squares(pw_tdoa, m_pos, c=340.29):
    """DOA estimation by minimizing TDOA error (L2 norm).

    The objective is formulated with the far-field assumption:

        minimize   : f(u) = |tau - m u|^2
        subject to : g(u) = |u|^2 - 1 = 0

    where u is the direction of arrival, tau is the vector of estimated
    difference of path (TDOA times sound speed), and m is the difference
    of microphone positions.

    Args:
        pw_tdoa : pairwise TDOA estimate, result from apkit.pairwise_tdoa.
        m_pos   : list of vector position in 3D space in meters.
        c       : (default 340.29 m/s) speed of sound.
    
    Returns:
        list of optimal solutions of u
    """
    pairs = pw_tdoa.keys()
    tau = np.matrix([pw_tdoa[p] * c for p in pairs]).T
    m = np.matrix([m_pos[i] - m_pos[j] for i, j in pairs])

    # dimension
    n, d = m.shape

    # check rank of m by computing svd
    u, s, vh = np.linalg.svd(m)
    rank = int((s > _TOLERANCE).sum())

    if rank == d:
        assert False    # TODO pepper has coplanar microphones
    else:
        assert rank == d - 1    # not able to compute rank deficiency of
                                # more than one
        sinv = np.diag(np.append(1. / s[:-1], 0))
        sinv = np.append(sinv, np.zeros((n-d, n-d)), axis=1)

        # psusdo inverse times tau, get min-norm solution of least-square
        # (without constraints)
        # the solution x must be orthognal to null space of m
        x = vh.H * sinv * u.H * tau

        if x.T * x <= 1.0:
            # in case of norm of x less than 1, add y in the null space to
            # make norm to one
            y = np.asscalar(1 - x.T * x) ** .5

            # two possible solutions
            u1 = x + vh[-1].H * y
            u2 = x - vh[-1].H * y
            return [u1.A1, u2.A1]
        else:
            # otherwise the solution must be somewhere f is not stationary,
            # while f - lambda * g is stationary, therefore
            # solve lambda (lagrange multiplier)
            lam = scipy.optimize.fsolve(_u_sqr_minus_1, 1e-5, (m, tau))
            u = np.linalg.solve(m.T * m - lam * np.eye(d), m.T * tau)

            # TODO: there are two solutions, one at mininum of f and
            # the other at maximum. However only one is solved here.
            # hack here, normalize unconstraint solution, and compare
            u2 = x / np.linalg.norm(x)

            if _error_func(u2, m, tau) < _error_func(u, m, tau):
                u = u2

            ''' tried with directly optimize objective numerically,
                but doesn't work
            res = scipy.optimize.minimize(_error_func, u2, (m, tau),
                    constraints={'type':'ineq', 'fun':_constraint_func})
            u3 = np.asmatrix(res.x).T
            print u3
            print np.linalg.norm(tau - m * u3)
            '''

            return [u.A1]

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

