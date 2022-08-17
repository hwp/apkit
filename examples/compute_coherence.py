#!/usr/bin/env python

import argparse

import numpy as np
import scipy.signal
import apkit

def _complex_coherence(x, y, win_size, fs):
    freq_bins, cxy = scipy.signal.csd(x, y, nperseg=win_size, fs=fs)
    _, cxx = scipy.signal.welch(x, nperseg=win_size, fs=fs)
    _, cyy = scipy.signal.welch(y, nperseg=win_size, fs=fs)
    coherence = cxy / np.sqrt(cxx.real * cyy.real)
    return freq_bins, coherence, cxx, cyy

def main(infile, outfile, win_size, plot_title, plot_max_freq):
    fs, sig = apkit.load_wav(infile)
    n_ch = sig.shape[0]
    for i in range(n_ch - 1):
        for j in range(i + 1, n_ch):
            freq_bins, coherence, psd_x, psd_y = _complex_coherence(sig[i], sig[j], 
                                                      win_size=win_size, fs=fs)
            with open(f'{outfile}_{i}vs{j}', 'w') as f:
                for freq, c, p, q in zip(freq_bins, coherence, psd_x, psd_y):
                    print(f'{freq}\t{c.real}\t{c.imag}\t{p}\t{q}', file=f)

    with open(f'{outfile}_plot.gp', 'w') as f:
        print('set terminal pngcairo size 1000,800', file=f)
        print(f'set xrange [0:{plot_max_freq}]', file=f)
        print('set grid', file=f)
        print('set samples 500', file=f)
        for i in range(n_ch - 1):
            for j in range(i + 1, n_ch):
                print(f'set title "{plot_title} -- ch{i} vs ch{j}"', file=f)
                print(f'set output "{outfile}_{i}vs{j}.png"', file=f)
                print('set multiplot layout 2,1', file=f)
                print('unset xlabel', file=f)
                print('set yrange [-.4:1]', file=f)
                print('set ylabel "Coherence"', file=f)
                print(f'plot "{outfile}_{i}vs{j}" u 1:2 smooth csplines w l lc rgb "#2000ff00" title "Real", '
                            '"" u 1:3 smooth csplines w l lc rgb "#20ff0000" title "Imaginary", '
                            '"" u 1:(sqrt($2**2+$3**2)) smooth csplines w l lc rgb "#200000ff" dt "." title "Magnitude"',
                      file=f)
                print('set title "PSD"', file=f)
                print('set xlabel "Frequency"', file=f)
                print('unset yrange', file=f)
                print('set ylabel "PSD (dB)"', file=f)
                print(f'plot "{outfile}_{i}vs{j}" u 1:(10.0*log10($4)) smooth csplines w l lc rgb "#2000ff00" title "ch{i}", '
                            f'"" u 1:(10.0*log10($5)) smooth csplines w l lc rgb "#20ff0000" title "ch{j}"',
                      file=f)
                print('unset multiplot', file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='print complex coherence')
    parser.add_argument('--win-size', '-w', type=int, required=True,
                        help='frame size')
    parser.add_argument('--outfile', '-o', type=str, required=True,
                        help='output files prefix')
    parser.add_argument('--plot-title', type=str, default='Coherence',
                        help='plot title')
    parser.add_argument('--plot-max-freq', type=int, default=1000,
                        help='max frequency to be plotted')
    parser.add_argument('infile', metavar='INFILE', type=str,
                        help='input audio file')
    args = parser.parse_args()
    main(args.infile, args.outfile, args.win_size, args.plot_title,
         args.plot_max_freq)

