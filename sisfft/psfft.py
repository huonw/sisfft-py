from __future__ import print_function
import numpy as np, scipy as sc
from numpy import fft
from scipy import optimize

from timer import timer
import logging

import naive, utils
from utils import NEG_INF, EPS

COST_RATIO = 1

OPT_BOUND = 1e10

def convolve(log_pmf1, log_pmf2, alpha, delta = None):
    # assert len(log_pmf1) == len(log_pmf2)
    if delta is not None:
        return _psfft_noshift(log_pmf1, log_pmf2, alpha, delta)
    else:
        # shift, convolve, unshift
        theta = _compute_theta(log_pmf1, log_pmf2)
        s1, log_mgf1 = utils.shift(log_pmf1, theta)
        s2, log_mgf2 = utils.shift(log_pmf2, theta)
        convolved = _psfft_noshift(s1, s2, alpha, NEG_INF)
        return utils.unshift(convolved, theta, (log_mgf1, 1), (log_mgf2, 1))

def convolve_square(log_pmf, alpha, delta = None):
    # TODO: make this more efficient
    return convolve(log_pmf, log_pmf, alpha, delta)

def _psfft_noshift(log_pmf1, log_pmf2, alpha, delta):
    true_conv_len, fft_conv_len = utils.pairwise_convolution_lengths(len(log_pmf1), len(log_pmf2))

    with timer('splitting'):
        splits1, normalisers1 = _split(log_pmf1, fft_conv_len,
                                       alpha, delta)
        splits2, normalisers2 = _split(log_pmf2, fft_conv_len,
                                       alpha, delta)

    nc_cost = len(log_pmf1) * len(log_pmf2)
    Q = max(len(log_pmf1), len(log_pmf2))
    psfft_cost = len(splits1) * len(splits2) * Q * np.log2(Q)
    scaled_cost = psfft_cost * COST_RATIO
    psfft_is_better = scaled_cost < nc_cost
    logging.debug('psFFT computed %s & %s splits, giving scaled cost %.2e (vs. NC cost %.2e). Using '
                  'psFFT? %s',
                  len(splits1), len(splits2),
                  scaled_cost,
                  nc_cost,
                  psfft_is_better)

    if not psfft_is_better:
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
                                           true_conv_len,
                                           fft_conv_len)
                accum = np.logaddexp(accum, conv)
    return accum

def _split(log_pmf, fft_conv_len, alpha, delta):
    sort_idx = np.argsort(log_pmf)
    raw_threshold = utils.error_threshold_factor(fft_conv_len) * alpha
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

def _filtered_mult_ifft(fft1, normaliser1, fft2, normaliser2,
                        true_conv_len, fft_conv_len):
    norm1 = np.linalg.norm(fft1)
    norm2 = np.linalg.norm(fft2)

    # norm1 & norm2 are sqrt(fft_conv_len) * norm(p) (resp. norm(q)),
    # i.e. too large by a factor of sqrt(fft_conv_len) each.
    threshold = utils.error_threshold_factor(fft_conv_len) * norm1 * norm2 / fft_conv_len

    entire_conv = fft.ifft(fft1 * fft2)[:true_conv_len]
    filtered = np.where(np.abs(entire_conv) > threshold,
                        np.real(entire_conv),
                        0.0)
    filtered = np.log(filtered)
    filtered += normaliser1 + normaliser2
    return filtered


def _compute_theta(log_pmf1, log_pmf2, Lnorm = 2):
    def f(theta):
        s1, _ = utils.shift(log_pmf1, theta)
        s2, _ = utils.shift(log_pmf2, theta)
        r = utils.log_dynamic_range(s1) * utils.log_dynamic_range(s2)
        return r
    return optimize.fminbound(f, -OPT_BOUND, OPT_BOUND)
