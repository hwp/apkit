#!/usr/bin/env python
"""
angular_spec_doa.py

Written by Weipeng He <heweipeng@gmail.com>
"""

import time  # debug
import sys  # debug
import math
import argparse

import numpy as np

import apkit

# Microphone 3D coordinates (unit is meter)
_MICROPHONE_COORDINATES = np.array([[-0.0267, 0.0343, 0.2066],
                                    [-0.0267, -0.0343, 0.2066],
                                    [0.0313, 0.0343, 0.2066],
                                    [0.0313, -0.0343, 0.2066]])

# Use signal up to 8 kHz for prediction
_MAX_FREQ = 8000


def load_ncov(path, win_size, hop_size):
    fs, sig = apkit.load_wav(path)
    nfbin = _MAX_FREQ * win_size // fs  # 0-8kHz
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
    tf = tf[:, :, :nfbin]
    return apkit.cov_matrix(tf)


def main(infile, outdir, afunc, win_size, hop_size, block_size, block_hop,
         min_sc):
    stime = time.time()

    # load candidate DOAs
    pts = apkit.load_pts_on_sphere()
    pts = pts[pts[:, 2] > -0.05]  # use upper half of the sphere
    # NOTE: alternatively use only points on the horizontal plane
    # pts = apkit.load_pts_horizontal(360)
    print('%.3fs: load points (%d)' % (time.time() - stime, len(pts)),
          file=sys.stderr)

    # compute neighbors (for peak finding)
    nlist = apkit.neighbor_list(pts, math.pi / 180.0 * 8.0)
    print('%.3fs: neighbor list' % (time.time() - stime), file=sys.stderr)

    # load signal
    fs, sig = apkit.load_wav(infile)
    print('%.3fs: load signal' % (time.time() - stime), file=sys.stderr)

    # compute delays (delay for each candidate DOA and each microphone)
    delays = apkit.compute_delay(_MICROPHONE_COORDINATES, pts, fs=fs)
    print('%.3fs: compute delays' % (time.time() - stime), file=sys.stderr)

    # compute empirical covariance matrix
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
    max_fbin = _MAX_FREQ * win_size // fs  # int
    assert max_fbin <= win_size // 2
    tf = tf[:, :, :max_fbin]  # 0-8kHz
    fbins = np.arange(max_fbin, dtype=float) / win_size
    if block_size is None:
        ecov = apkit.empirical_cov_mat(tf)
    else:
        ecov = apkit.empirical_cov_mat_by_block(tf, block_size, block_hop)
    nch, _, nblock, nfbin = ecov.shape
    print('%.3fs: empirical cov matrix (nfbin=%d)' %
          (time.time() - stime, nfbin),
          file=sys.stderr)

    # local angular spectrum function
    phi = afunc(ecov, delays, fbins)
    print('%.3fs: compute phi' % (time.time() - stime), file=sys.stderr)

    # find local maxima
    lmax = apkit.local_maxima(phi, nlist, th_phi=min_sc)
    print('%.3fs: find local maxima' % (time.time() - stime), file=sys.stderr)

    # merge predictions that have similar azimuth predicitons
    # NOTE: skip this step if the candinate DOAs are on the horizontal plane
    lmax = apkit.merge_lm_on_azimuth(phi, lmax, pts, math.pi / 180.0 * 5.0)
    print('%.3fs: refine local maxima' % (time.time() - stime),
          file=sys.stderr)

    # save results
    # each file contains the predicted angular spectrum for each frame/block
    # each line has five tokens:
    #   (1) x coordinate of the candidate DOA
    #   (2) y coordinate of the candidate DOA
    #   (3) z coordinate of the candidate DOA
    #   (4) angular spectrum value
    #   (5) 1 if this is a local maximum, otherwise 0
    for t in range(nblock):
        with open(f'{outdir}/{t:06d}', 'w') as f:
            for i in range(len(pts)):
                print('%g %g %g %g %d' % (pts[i, 0], pts[i, 1], pts[i, 2],
                                          phi[i, t], 1 if i in lmax[t] else 0),
                      file=f)
    print('%.3fs: save results' % (time.time() - stime), file=sys.stderr)


_FUNCTIONS = {
    'mvdr': apkit.phi_mvdr,
    'mvdr-snr': apkit.phi_mvdr_snr,
    'srp-phat': apkit.phi_srp_phat,
    'srp-nonlin': apkit.phi_srp_phat_nonlin,
    'sevd-music': apkit.sevd_music
}

_FUNCS_WITH_NCOV = {
    'mvdr-ncov': apkit.MVDR_NCOV,
    'mvdr-ncov-snr': apkit.MVDR_NCOV_SNR,
    'mvdr-ncov-sig': apkit.MVDR_NCOV_SIG,
    'music': apkit.MUSIC,
    'gsvd-music': apkit.GSVD_MUSIC
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DOA estimation using angular spectrum methods')
    parser.add_argument('infile',
                        metavar='INPUT_FILE',
                        type=argparse.FileType('rb'),
                        help='input wav file')
    parser.add_argument('outdir',
                        metavar='OUTPUT_DIR',
                        type=str,
                        help='output directory')
    parser.add_argument('-w',
                        '--window-size',
                        metavar='WIN_SIZE',
                        type=int,
                        default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument('-o',
                        '--hop-size',
                        metavar='HOP_SIZE',
                        type=int,
                        default=1024,
                        help='(default 1024) hop size, '
                        'number of samples between windows')
    parser.add_argument('--block-size',
                        metavar='BLOCK_SIZE',
                        type=int,
                        default=7,
                        help='(default 7) if not None, compute '
                        'spatial covariance matrix in blocks, each include '
                        'block_size frames.')
    parser.add_argument('--block-hop',
                        metavar='BLOCK_HOP',
                        type=int,
                        default=4,
                        help='(default 4) used with block_size, '
                        'number of frames between consecutive blocks.')
    parser.add_argument('-f',
                        '--function',
                        metavar='FUNC',
                        choices=list(_FUNCTIONS.keys()) +
                        list(_FUNCS_WITH_NCOV.keys()),
                        required=True,
                        help='local angular spectrum function')
    parser.add_argument('-n',
                        '--noise',
                        metavar='NOISE_FILE',
                        type=argparse.FileType('rb'),
                        default=None,
                        help='(default None) sample background noise file')
    parser.add_argument('--min-score',
                        metavar='SCORE',
                        type=float,
                        default=0.0,
                        help='(default 0.0) minimun score for peaks')
    args = parser.parse_args()
    if args.function in _FUNCTIONS:
        func = _FUNCTIONS[args.function]
    elif args.function in _FUNCS_WITH_NCOV:
        ncov = load_ncov(args.noise, args.window_size, args.hop_size)
        func = _FUNCS_WITH_NCOV[args.function](ncov)
        args.noise.close()
    else:
        raise KeyError
    main(args.infile, args.outdir, func, args.window_size, args.hop_size,
         args.block_size, args.block_hop, args.min_score)
    args.infile.close()

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
