Audio Processing Kit (apkit)
============================

This is python library for audio processing functions, including:

* Basic I/O
* Basic Operations
* Cross-correlation and TDOA estimation
* Spatial spectral-based sound source localization
* Beamforming

Dependency
----------

* Numpy
* Scipy

Conventions
-----------

* Arrays are numpy arrays.
* Multi-dimension array indices follow such order (if exist): channel, time, frequency.

Install
-------

```
pip install -e .
```

Documentation
-------------

The functions of this library are listed (not exhaustively) in the following tables, separated by their categories. For the usage of the functions, please read the doc in the code or use `help(apkit.FUNCTION)` in python.

### Basic I/O

| Functions       | Description                                          |
| --------------- | ---------------------------------------------------- |
| `load_wav`      | Load wav file                                        |
| `load_metadata` | Load metadata of a wav file without reading its data |
| `save_wav`      | Store audio to a wav file                            |

### Basic Operations

| Functions             | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `stft`                | Short time Fourier Transform                         |
| `istft`               | Inverse short time Fourier Transform                 |
| `cola_hamming`        | Constant-Overlap-Add hamming window                  |
| `cola_rectangle`      | Constant-Overlap-Add rectangle window                |
| `freq_upsample`       | Upsampling in frequency domain by padding            |
| `power`               | Compute power of time domain signal                  |
| `power_tf`            | Compute power of TF domain signal                    |
| `snr`                 | Compute signal-to-noise ratio                        |
| `steering_vector`     | Compute steering vector from delay                   |
| `compute_delay`       | Compute delays to microphones given DOA              |
| `neighbor_list`       | List of neighbor DOAs                                |
| `angular_distance`    | Angular distance                                     |
| `azimuth_distance`    | Azimuth distance                                     |
| `load_pts_horizontal` | Load evenly distributed DOAs in the horizontal plane |
| `load_pts_on_sphere`  | Load "almost" evenly distributed DOAs on the sphere  |

### Covariance Matrix

| Functions                    | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `cov_matrix`                 | Covariance matrix                               |
| `empirical_cov_mat`          | Covariance matrix at different frames           |
| `empirical_cov_mat_by_block` | Covariance matrix at different blocks of frames |

### Cross-correlation and TDOA

| Functions           | Description       |
| ------------------- | ----------------- |
| `gcc_phat`          | GCC-PHAT          |
| `cross_correlation` | Cross-correlation |

### Beamforming

| Functions             | Description                     |
| --------------------- | ------------------------------- |
| `apply_beamforming`   | Apply beamforming given weights |
| `bf_delay_sum`        | Delay sum beamformer            |
| `bf_weight_delay_sum` | Delay sum beamformer weights    |

> **Note**
> There was a "static" version of MVDR beamformer, which is not "adaptive" as it is supposed to be.

### Spatial Spectrum-based DOA estimation

| Functions             | Description                                 |
| --------------------- | ------------------------------------------- |
| `phi_srp_phat`        | SRP-PHAT                                    |
| `phi_srp_phat_nonlin` | SPR-PHAT with non-linear correction         |
| `phi_mvdr_snr`        | MVDR for SSL with SNR scoring               |
| `sevd_music`          | MUSIC (vanilla, white noise)                |
| `MUSIC`               | MUSIC (known noise covariance)              |
| `GSVD_MUSIC`          | GSVD-MUSIC (singular value decomposition)   |
| `local_maxima`        | Find local maxima in spatial spectrogram    |
| `merge_lm_on_azimuth` | Merge local maxima with same azimuth        |
| `vec2ae`              | Unit vector to azimuth-elevation conversion |

