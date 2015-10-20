from __future__ import print_function
import numpy as np, scipy as sc
from numpy import fft
from scipy import optimize

from timer import timer
import logging

import naive, utils
from utils import NEG_INF, EPS

COST_RATIO = 0.5
# when we square we do approximately half the work of a general
# convolution.
COST_RATIO_SQUARE = COST_RATIO * 0.5

OPT_BOUND = 1e10

def convolve(log_pmf1, log_pmf2, alpha, delta = None):
    # assert len(log_pmf1) == len(log_pmf2)
    if delta is not None:
        return _psfft_noshift(log_pmf1, log_pmf2, alpha, delta,
                              pairwise = True,
                              square_1 = False)[0]
    else:
        # shift, convolve, unshift
        theta = _compute_theta(log_pmf1, log_pmf2)
        s1, log_mgf1 = utils.shift(log_pmf1, theta)
        s2, log_mgf2 = utils.shift(log_pmf2, theta)
        convolved = _psfft_noshift(s1, s2, alpha, NEG_INF)[0]
        return utils.unshift(convolved, theta, (log_mgf1, 1), (log_mgf2, 1))

def convolve_square(log_pmf, alpha, delta = None):
    if delta is None:
        return convolve(log_pmf, log_pmf, alpha, delta)
    else:
        return _psfft_noshift(log_pmf, np.array([]), alpha, delta,
                              pairwise = False,
                              square_1 = True)[1]

# computes `log_pmf1 * log_pmf2, log_pmf1 * log_pmf1`
def convolve_and_square(log_pmf1, log_pmf2, alpha, delta = None):
    if delta is None:
        # not sure if this is the best approach in general, probably
        # better to have some heuristics about choosing to shift
        # together vs. individually or something
        co = convolve(log_pmf1, log_pmf2, alpha, None)
        sq = convolve_square(log_pmf1, alpha, None)
        return co, sq
    else:
        return _psfft_noshift(log_pmf1, log_pmf2, alpha, delta,
                              pairwise = True, square_1 = True)

def _psfft_noshift(log_pmf1, log_pmf2, alpha, delta,
                   pairwise, square_1):
    """This function does some sort of pairwise convolution with its arguments, it has three modes:
    - pairwise = True: log_pmf1 * log_pmf2
    - square_1 = True: log_pmf1 * log_pmf1
    - both of those at once

    The square_1 case computes the square more efficiently than using
    pairwise = True with log_pmf2 = log_pmf1, and specifying both at
    once may be able to share computations to compute both log_pmf1 *
    log_pmf2, log_pmf1 * log_pmf1 faster than individually.

    Specifically, pairwise = square_1 = True can share some things
    when the FFT-lengths of the two convolutions are the same. This
    actually occurs surprisingly often in the context of sisFFT, where
    one is convolving-by-squaring, i.e. building up an accumulation
    vector by convolving it with progressively longer vectors.
    """
    # we should do *something*
    assert pairwise or square_1

    true_conv_len, fft_conv_len = utils.pairwise_convolution_lengths(len(log_pmf1), len(log_pmf2))
    true_conv_len_sq, fft_conv_len_sq = utils.pairwise_convolution_lengths(len(log_pmf1),
                                                                           len(log_pmf1))

    can_reuse_pairwise = pairwise and fft_conv_len == fft_conv_len_sq
    len1 = len(log_pmf1)
    len2 = len(log_pmf2)

    with timer('splitting'):
        if pairwise:
            splits1, normalisers1 = _split(log_pmf1, fft_conv_len,
                                           alpha, delta,
                                           _split_limit(len1, len2, None, COST_RATIO))
            if splits1 is not None:
                splits2, normalisers2 = _split(log_pmf2, fft_conv_len,
                                               alpha, delta,
                                               _split_limit(len1, len2, len(splits1), COST_RATIO))
            else:
                splits2, normalisers2 = None, None

        if can_reuse_pairwise:
            splits1_sq, normalisers1_sq = splits1, normalisers1
        else:
            # different numbers so we need to re-split
            splits1_sq, normalisers1_sq = _split(log_pmf1, fft_conv_len_sq,
                                                 alpha, delta,
                                                 _split_limit(len1, None, None, COST_RATIO))


    nc = nc_sq = None

    if pairwise:
        nc_is_better = _is_nc_faster(len(log_pmf1), splits1,
                                     len(log_pmf2), splits2,
                                     COST_RATIO)
        if nc_is_better:
            with timer('naive'):
                nc = naive.convolve_naive(log_pmf1, log_pmf2)
    if square_1:
        nc_is_better_sq = _is_nc_faster(len(log_pmf1), splits1_sq,
                                        len(log_pmf1), splits1_sq,
                                        COST_RATIO_SQUARE)
        if nc_is_better_sq:
            with timer('naive'):
                nc_sq = naive.convolve_naive(log_pmf1, log_pmf1)

    need_to_pairwise = nc is None and pairwise
    need_to_square = nc_sq is None and square_1
    # we can only reuse it if it is actually computed
    can_reuse_pairwise &= need_to_pairwise

    with timer('ffts'):
        if need_to_pairwise:
            ffts1 = fft.fft(np.exp(splits1), n = fft_conv_len, axis = 1)
            ffts2 = fft.fft(np.exp(splits2), n = fft_conv_len, axis = 1)

        if need_to_square:
            if can_reuse_pairwise:
                ffts1_sq = ffts1
            else:
                ffts1_sq = fft.fft(np.exp(splits1_sq),
                                   n = fft_conv_len_sq, axis = 1)


    if need_to_pairwise:
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
    else:
        accum = nc

    if need_to_square:
        accum_sq = np.repeat(NEG_INF, true_conv_len_sq)

        with timer('defft-square'):
            for i, normaliser1 in enumerate(normalisers1_sq):
                fft1 = ffts1_sq[i, :]
                conv_self = _filtered_mult_ifft(fft1, normaliser1,
                                                fft1, normaliser1,
                                                true_conv_len_sq,
                                                fft_conv_len_sq)
                accum_sq = np.logaddexp(accum_sq, conv_self)
                for j in range(i + 1, len(normalisers1_sq)):
                    normaliser2 = normalisers1_sq[j]
                    fft2 = ffts1_sq[j, :]
                    conv = _filtered_mult_ifft(fft1, normaliser1,
                                               fft2, normaliser2,
                                               true_conv_len_sq,
                                               fft_conv_len_sq)
                    # this is counted twice (for (i, j) and (j, i)),
                    # so we need to double
                    conv += np.log(2)
                    accum_sq = np.logaddexp(accum_sq, conv)
    else:
        accum_sq = nc_sq

    if pairwise: assert len(accum) == true_conv_len
    if square_1: assert len(accum_sq) == true_conv_len_sq
    return accum, accum_sq

def _split(log_pmf, fft_conv_len, alpha, delta,
           split_limit):
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
    for i, idx in enumerate(sort_idx[::-1]):
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

            if len(splits) + 1 > split_limit:
                logging.debug('bailing out of splits after %s elements (%s splits)',
                              i + 1, len(splits) + 1)
                return None, None

        current_split[idx] = log_val - log_current_max
    splits.append(current_split)
    return np.array(splits), np.array(maxes)

def _split_limit(len1, len2, splits1, cost_ratio):
    nc_cost = len1 * len1 if len2 is None else len1 * len2
    Q = max(len1, len2)
    split_ratio = nc_cost / (Q * np.log2(Q) * cost_ratio)
    if len2 is None:
        # we're squaring, so both sides have the same number of splits
        lim = np.sqrt(split_ratio)
    else:
        # worst case the RHS will be 1
        other_split = 1 if splits1 is None else splits1
        lim = split_ratio / other_split
    logging.debug('computed split limit for %s %s %s %s as %s',
                  len1, len2, splits1, cost_ratio, lim)
    return lim

def _is_nc_faster(len1, splits1,
                  len2, splits2,
                  cost_ratio):
    nc_cost = len1 * len2
    Q = max(len1, len2)

    if splits1 is None or splits2 is None:
        nc_is_better = True
        logging.debug('psFFT didn\'t compute all splits (%s, %s). Using psFFT? False',
                      splits1, splits2)
    else:
        psfft_cost = len(splits1) * len(splits2) * Q * np.log2(Q)
        scaled_cost = psfft_cost * cost_ratio
        nc_is_better = scaled_cost > nc_cost

        logging.debug('psFFT computed %s & %s splits, giving scaled cost %.2e (vs. NC cost %.2e). '
                       'Using psFFT? %s',
                      len(splits1), len(splits2),
                      scaled_cost,
                      nc_cost,
                      not nc_is_better)
    return nc_is_better


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
