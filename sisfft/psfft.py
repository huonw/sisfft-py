from __future__ import print_function
import numpy as np, scipy as sc
from numpy import fft
from scipy import optimize

from timer import timer
import unittest

import naive


EPS = np.finfo(float).eps
NEG_INF = float('-inf')

COST_RATIO = 1e-10

def convolve(log_pmf1, log_pmf2, alpha, delta = None):
    # assert len(log_pmf1) == len(log_pmf2)
    if delta is not None:
        return _psfft_noshift(log_pmf1, log_pmf2, alpha, delta)
    else:
        # shift, convolve, unshift
        raise NotImplementedError()

def convolve_square(log_pmf, alpha, delta = None):
    # TODO: make this more efficient
    return convolve(log_pmf, log_pmf, alpha, delta)

def _psfft_noshift(log_pmf1, log_pmf2, alpha, delta):
    true_conv_len, fft_conv_len = _convolution_lengths(len(log_pmf1), len(log_pmf2))

    with timer('splitting'):
        splits1, normalisers1 = _split(log_pmf1, true_conv_len,
                                       alpha, delta)
        splits2, normalisers2 = _split(log_pmf2, true_conv_len,
                                       alpha, delta)

    nc_cost = len(log_pmf1) * len(log_pmf2)
    Q = max(len(log_pmf1), len(log_pmf2))
    psfft_cost = len(splits1) * len(splits2) * Q * np.log2(Q)

    if psfft_cost * COST_RATIO > nc_cost:
        with timer('naive'):
            return naive.convolve_naive(log_pmf1, log_pmf2)

    with timer('ffts'):
        ffts1 = fft.fft(np.exp(splits1), n = fft_conv_len, axis = 1)
        ffts2 = fft.fft(np.exp(splits2), n = fft_conv_len, axis = 1)

    accum = np.repeat(NEG_INF, true_conv_len)

    with timer('defft'):
        for i, normaliser1 in enumerate(normalisers1):
            fft1 = ffts1[i, :]
            for j, normaliser2 in enumerate(normalisers2):
                fft2 = ffts2[j, :]
                conv = _filtered_mult_ifft(fft1, normaliser1,
                                           fft2, normaliser2,
                                           true_conv_len)
                accum = np.logaddexp(accum, conv)
    return accum

def _threshold_factor(conv_len):
    if conv_len > 2**5:
        c = 13.5
    else:
        c = 16
    return EPS * c * np.log2(conv_len)

def _split(log_pmf, conv_len, alpha, delta):
    sort_idx = np.argsort(log_pmf)
    raw_threshold = _threshold_factor(conv_len) * alpha
    log_threshold = -np.log(raw_threshold) / 2.0
    log_delta = np.log(delta)

    log_current_max = log_pmf[sort_idx[-1]]
    log_current_norm_sq = NEG_INF

    def new_split():
        return np.repeat(NEG_INF, len(log_pmf))
    current_split = new_split()

    maxes = [log_current_max]
    splits = []
    for idx in sort_idx[::-1]:
        log_val = log_pmf[idx]
        if log_val <= log_delta:
            break
        log_next_norm_sq = np.logaddexp(log_current_norm_sq, log_val * 2.0)
        r = log_next_norm_sq / 2.0 - log_val
        if r < log_threshold:
            log_current_norm_sq = log_next_norm_sq
        else:
            # threshold violated so this point needs to be in a new
            # split
            splits.append(current_split)
            current_split = new_split()

            maxes.append(log_val)
            log_current_norm_sq = log_val * 2.0
            log_current_max = log_val

        current_split[idx] = log_val - log_current_max
    splits.append(current_split)
    return np.array(splits), np.array(maxes)

def _next_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))

def _convolution_lengths(a, b):
    true = a + b - 1
    return true, _next_power_of_two(true)

def _filtered_mult_ifft(fft1, normaliser1, fft2, normaliser2, true_conv_len):
    norm1 = np.linalg.norm(fft1)
    norm2 = np.linalg.norm(fft2)

    threshold = _threshold_factor(true_conv_len) * norm1 * norm2

    entire_conv = fft.ifft(fft1 * fft2)[:true_conv_len]
    # TODO: possibly need threshold * np.max(np.abs(entire_conv))?
    filtered = np.where(np.abs(entire_conv) > threshold,
                        np.log(np.real(entire_conv)),
                        NEG_INF)
    filtered += normaliser1 + normaliser2
    return filtered
