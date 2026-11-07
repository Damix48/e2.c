"""Mel filterbank matching transformers.audio_utils.mel_filter_bank (HTK, norm=None)."""
import numpy as np


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filters(num_freq_bins, num_mel, min_freq, max_freq, sampling_rate):
    all_freqs = np.linspace(0, sampling_rate // 2, num_freq_bins)
    m_min = _hz_to_mel(min_freq)
    m_max = _hz_to_mel(max_freq)
    m_pts = np.linspace(m_min, m_max, num_mel + 2)
    f_pts = _mel_to_hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    return fb.astype(np.float32)


def hann_window(n):
    # periodic Hann, matching signal.hann_window / window_function
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)
