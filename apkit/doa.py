"""
doa.py

Direction of arrival (DOA) estimation.

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import os
import math

import numpy as np
import scipy

_TOLERANCE = 1e-13

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

def load_pts_horizontal(npts=360):
    """Load points evenly distributed on the unit circle on x-y plane

    Args:
        npts : (default 360) number of points

    Returns:
        pts  : array of points on a unit circle
    """
    aindex = np.arange(npts) * 2 * np.pi / npts
    return np.array([np.cos(aindex), np.sin(aindex), np.zeros(npts)]).T

def neighbor_list(pts, dist, scale_z=1.0):
    """List of neighbors (using angular distance as metic)

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
    for i in range(len(pts)):
        amat[i,i] = False

    # convert to list
    return [list(np.nonzero(n)[0]) for n in amat]

_norm = np.linalg.norm

def angular_distance(a, b):
    denom = (_norm(a) * _norm(b))
    if denom < 1e-16:
        return math.pi
    sim = np.dot(a, b) / denom
    if sim > 1.0:
        return 0.0
    elif sim < -1.0:
        return math.pi
    else:
        return math.acos(sim)

def azimuth_distance(a, b):
    return angular_distance(a[:2], b[:2])

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
        print(e)
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
    pairs = list(pw_tdoa.keys())
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

