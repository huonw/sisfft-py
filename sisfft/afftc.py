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

OPT_BOUND = 1e4

ESTIMATE_ONE_SPLIT = 1
ESTIMATE_TWO_SPLITS = 2

# we use beta because it is the more natural parameter, but it does
# differ to the paper, and to the rest of the code (we didn't want to
# unnecessarily risk bugs by changing it)
def convolve(log_pmf1, log_pmf2, beta, delta = None,
             enable_fast_path = True):
    """Compute $log(exp(log_pmf1) * exp(log_pmf2))$, by trimming then
       convolving (if delta is not None)."""
    # assert len(log_pmf1) == len(log_pmf2)
    if delta is not None:
        return _afftc_noshift(log_pmf1, log_pmf2, 1/beta, delta,
                              pairwise = True,
                              square_1 = False,
                              enable_fast_path = enable_fast_path)[0]
    else:
        return _convolve_no_lower_bound(log_pmf1, log_pmf2, beta,
                                        enable_fast_path = enable_fast_path)

def convolve_square(log_pmf, beta, delta = None):
    if delta is None:
        return convolve(log_pmf, log_pmf, beta, delta)
    else:
        return _afftc_noshift(log_pmf, np.array([]), beta, delta,
                              pairwise = False,
                              square_1 = True,
                              enable_fast_path = True)[1]

# computes `log_pmf1 * log_pmf2, log_pmf1 * log_pmf1`
def convolve_and_square(log_pmf1, log_pmf2, beta, delta = None):
    if delta is None:
        # not sure if this is the best approach in general, probably
        # better to have some heuristics about choosing to shift
        # together vs. individually or something
        co = convolve(log_pmf1, log_pmf2, beta, None)
        sq = convolve_square(log_pmf1, beta, None)
        return co, sq
    else:
        return _afftc_noshift(log_pmf1, log_pmf2, beta, delta,
                              pairwise = True, square_1 = True,
                              enable_fast_path = True)

def _convolve_no_lower_bound(log_pmf1, log_pmf2, beta,
                             enable_fast_path):
    alpha = 1.0 / beta
    if enable_fast_path:
        true_conv_len, fft_conv_len = utils.pairwise_convolution_lengths(len(log_pmf1),
                                                                         len(log_pmf2))

        pmf1 = np.exp(log_pmf1)
        fft1 = fft.fft(pmf1, n = fft_conv_len)
        direct, bad_places = _direct_fft_conv(log_pmf1, pmf1, fft1, log_pmf2,
                                              true_conv_len, fft_conv_len,
                                              alpha, NEG_INF)

        used_nc = _use_nc_if_better(log_pmf1, ESTIMATE_ONE_SPLIT,
                                    log_pmf2, ESTIMATE_TWO_SPLITS,
                                    direct, bad_places,
                                    COST_RATIO)
        if used_nc:
            #logging.debug('convolved without lower bound without shifting')
            return direct

    # shift, convolve, unshift
    theta = _compute_theta(log_pmf1, log_pmf2)
    s1, log_mgf1 = utils.shift(log_pmf1, theta)
    s2, log_mgf2 = utils.shift(log_pmf2, theta)
    convolved = _afftc_noshift(s1, s2, beta, NEG_INF,
                               pairwise = True,
                               square_1 = False,
                               enable_fast_path = enable_fast_path)[0]
    return utils.unshift(convolved, theta, (log_mgf1, 1), (log_mgf2, 1))

def checked_fftc(log_pmf1, log_pmf2, alpha):
    true_conv_len, fft_conv_len = utils.pairwise_convolution_lengths(len(log_pmf1),
                                                                     len(log_pmf2))
    pmf1 = np.exp(log_pmf1)
    fft1 = fft.fft(pmf1, n = fft_conv_len)
    return _direct_fft_conv(log_pmf1, pmf1, fft1, log_pmf2,
                            true_conv_len, fft_conv_len,
                            alpha, NEG_INF)


def _afftc_noshift(log_pmf1, log_pmf2, beta, delta,
                   pairwise, square_1,
                   enable_fast_path):
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
    alpha = 1.0 / beta

    true_conv_len, fft_conv_len = utils.pairwise_convolution_lengths(len(log_pmf1), len(log_pmf2))
    true_conv_len_sq, fft_conv_len_sq = utils.pairwise_convolution_lengths(len(log_pmf1),
                                                                           len(log_pmf1))

    can_reuse_pairwise = pairwise and fft_conv_len == fft_conv_len_sq
    len1 = len(log_pmf1)
    len2 = len(log_pmf2)

    answer = answer_sq = None

    if enable_fast_path:
        with timer('initial fft'):
            pmf1 = np.exp(log_pmf1)
            fft1 = fft.fft(pmf1, n = fft_conv_len)
            if pairwise:
                direct, bad_places = _direct_fft_conv(log_pmf1, pmf1, fft1, log_pmf2,
                                                      true_conv_len, fft_conv_len,
                                                      alpha, delta)

            if square_1:
                if can_reuse_pairwise:
                    direct_sq, bad_places_sq = _direct_fft_conv(log_pmf1, pmf1, fft1, None,
                                                                true_conv_len_sq, fft_conv_len_sq,
                                                                alpha, delta)
                else:
                    fft1_sq = fft.fft(pmf1, n = fft_conv_len_sq)
                    direct_sq, bad_places_sq = _direct_fft_conv(log_pmf1, pmf1, fft1_sq, None,
                                                                true_conv_len_sq, fft_conv_len_sq,
                                                                alpha, delta)
        if pairwise:
            used_nc = _use_nc_if_better(log_pmf1, ESTIMATE_ONE_SPLIT,
                                        log_pmf2, ESTIMATE_TWO_SPLITS,
                                        direct, bad_places,
                                        COST_RATIO)
            if used_nc:
                answer = direct

        if square_1:
            used_nc = _use_nc_if_better(log_pmf1, ESTIMATE_TWO_SPLITS,
                                        log_pmf1, ESTIMATE_TWO_SPLITS,
                                        direct_sq, bad_places_sq,
                                        COST_RATIO_SQUARE)
            if used_nc:
                answer_sq = direct_sq

    need_to_pairwise = answer is None and pairwise
    need_to_square = answer_sq is None and square_1
    # we can only reuse it if it is actually computed
    can_reuse_pairwise &= need_to_pairwise

    with timer('split maxima'):
        if need_to_pairwise:
            if enable_fast_path:
                limit = _split_limit(len1, len2, None,
                                     len(bad_places),
                                     COST_RATIO)
            else:
                limit = -NEG_INF

            maxima1 = _split_maxima(log_pmf1, fft_conv_len,
                                     alpha, delta,
                                    limit)
            if maxima1 is not None:
                if enable_fast_path:
                    limit = _split_limit(len1, len2,
                                         len(maxima1),
                                         len(bad_places),
                                         COST_RATIO)
                else:
                    limit = -NEG_INF

                maxima2 = _split_maxima(log_pmf2, fft_conv_len,
                                        alpha, delta,
                                        limit)
            else:
                maxima2 = None, None

        if need_to_square:
            if can_reuse_pairwise:
                maxima1_sq = maxima1
            else:
                # different numbers so we need to re-split
                if enable_fast_path:
                    limit = _split_limit(len1, None, None,
                                         len(bad_places_sq),
                                         COST_RATIO_SQUARE)
                else:
                    limit = -NEG_INF
                maxima1_sq = _split_maxima(log_pmf1, fft_conv_len_sq,
                                           alpha, delta,
                                           limit)

    if need_to_pairwise and enable_fast_path:
        used_nc = _use_nc_if_better(log_pmf1, maxima1,
                                    log_pmf2, maxima2,
                                    direct, bad_places,
                                    COST_RATIO)
        if used_nc:
            answer = direct
    if need_to_square and enable_fast_path:
        used_nc = _use_nc_if_better(log_pmf1, maxima1_sq,
                                    log_pmf1, maxima1_sq,
                                    direct_sq, bad_places_sq,
                                    COST_RATIO_SQUARE)
        if used_nc:
            answer_sq = direct_sq

    need_to_pairwise &= answer is None
    need_to_square &= answer_sq is None
    # we can only reuse it if it is actually computed
    can_reuse_pairwise &= need_to_pairwise

    with timer('actual splits'):
        if need_to_pairwise:
            splits1 = _splits_from_maxima(log_pmf1, maxima1, delta)
            splits2 = _splits_from_maxima(log_pmf2, maxima2, delta)

        if need_to_square:
            if can_reuse_pairwise:
                splits1_sq = splits1
            else:
                splits1_sq = _splits_from_maxima(log_pmf1, maxima1_sq, delta)

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
            for i, normaliser1 in enumerate(maxima1):
                fft1 = ffts1[i, :]
                for j, normaliser2 in enumerate(maxima2):
                    fft2 = ffts2[j, :]
                    conv = _filtered_mult_ifft(fft1, normaliser1,
                                               fft2, normaliser2,
                                               true_conv_len,
                                               fft_conv_len)
                    accum = np.logaddexp(accum, conv)
        answer = accum

    if need_to_square:
        accum_sq = np.repeat(NEG_INF, true_conv_len_sq)

        with timer('defft-square'):
            for i, normaliser1 in enumerate(maxima1_sq):
                fft1 = ffts1_sq[i, :]
                conv_self = _filtered_mult_ifft(fft1, normaliser1,
                                                fft1, normaliser1,
                                                true_conv_len_sq,
                                                fft_conv_len_sq)
                accum_sq = np.logaddexp(accum_sq, conv_self)
                for j in range(i + 1, len(maxima1_sq)):
                    normaliser2 = maxima1_sq[j]
                    fft2 = ffts1_sq[j, :]
                    conv = _filtered_mult_ifft(fft1, normaliser1,
                                               fft2, normaliser2,
                                               true_conv_len_sq,
                                               fft_conv_len_sq)
                    # this is counted twice (for (i, j) and (j, i)),
                    # so we need to double
                    conv += np.log(2)
                    accum_sq = np.logaddexp(accum_sq, conv)
        answer_sq = accum_sq

    if pairwise: assert len(answer) == true_conv_len
    if square_1: assert len(answer_sq) == true_conv_len_sq
    return answer, answer_sq

def _direct_fft_conv(log_pmf1, pmf1, fft1, log_pmf2, true_conv_len, fft_conv_len,
                     alpha, delta):
    if log_pmf2 is None:
        norms = np.linalg.norm(pmf1)**2
        fft_conv = fft1**2
    else:
        pmf2 = np.exp(log_pmf2)
        fft2 = fft.fft(pmf2, n = fft_conv_len)
        norms = np.linalg.norm(pmf1) * np.linalg.norm(pmf2)
        fft_conv = fft1 * fft2

    raw_conv = np.abs(fft.ifft(fft_conv)[:true_conv_len])
    error_level = utils.error_threshold_factor(fft_conv_len) * norms
    threshold = error_level * (alpha + 1)
    places_of_interest = raw_conv <= threshold

    if delta > error_level:
        ignore_level = delta - error_level

        conv_to_ignore = raw_conv < ignore_level
        raw_conv[conv_to_ignore] = 0.0
        places_of_interest &= ~conv_to_ignore

    log_conv = np.log(raw_conv)

    # the convolution is totally right, already: no need to consult
    # the support
    if not places_of_interest.any():
        return log_conv, []

    Q = 2 * len(log_pmf1) - 1 if log_pmf2 is None else len(log_pmf1) + len(log_pmf2) - 1

    support1 = log_pmf1 > NEG_INF
    if log_pmf2 is None:
        support2 = np.array([])
    else:
        support2 = log_pmf2 > NEG_INF
    if Q <= 2**42 and (not support1.all() or not support2.all()):
        # quickly compute the support; this is accurate given the
        # bound (if the vectors have no zeros, then the convolution
        # has no zeros, so we can skip this)
        fft_support1 = fft.fft(support1, n = fft_conv_len)
        if log_pmf2 is None:
            fft_support = fft_support1**2
        else:
            support2 = log_pmf2 > NEG_INF
            fft_support2 = fft.fft(support2, n = fft_conv_len)
            fft_support = fft_support1 * fft_support2

        raw_support = np.abs(fft.ifft(fft_support)[:true_conv_len])
        threshold = utils.error_threshold_factor(fft_conv_len) * 2 * Q
        zeros = raw_support <= threshold
        log_conv[zeros] = NEG_INF
        bad_places = np.where(~zeros & places_of_interest)[0]
    else:
        # can't/don't need to compute the support accurately.
        bad_places = np.where(places_of_interest)[0]

    return log_conv, bad_places

def _split_maxima(log_pmf, fft_conv_len, alpha, delta,
                  split_limit):
    sort_idx = np.argsort(log_pmf)
    raw_threshold = utils.error_threshold_factor(fft_conv_len) * alpha
    log_threshold = -np.log(raw_threshold) / 2.0
    log_delta = np.log(delta)

    log_current_max = log_pmf[sort_idx[-1]]
    log_current_norm_sq = NEG_INF

    maxes = [log_current_max]
    for i, idx in enumerate(sort_idx[::-1]):
        log_val = log_pmf[idx]
        if log_val < log_delta:
            break
        log_next_norm_sq = np.logaddexp(log_current_norm_sq, log_val * 2.0)
        r = log_next_norm_sq / 2.0 - log_val
        if r < log_threshold:
            log_current_norm_sq = log_next_norm_sq
        else:
            maxes.append(log_val)
            log_current_norm_sq = log_val * 2.0
            log_current_max = log_val

            if len(maxes) > split_limit:
                logging.debug('bailing out of splits after %s elements (%s splits)',
                              i + 1, len(maxes))
                return None

    return np.array(maxes)

def _splits_from_maxima(log_pmf, split_maxima, delta):
    log_delta = np.log(delta)
    pmf_len = len(log_pmf)
    num_splits = len(split_maxima)
    splits = np.full((num_splits, pmf_len), NEG_INF)

    for i in range(num_splits):
        if i == num_splits - 1:
            # make sure we include the boundary itself
            lo = np.nextafter(log_delta, NEG_INF)
        else:
            lo = split_maxima[i + 1]
        hi = split_maxima[i]

        elems = (lo < log_pmf) & (log_pmf <= hi)
        np.copyto(splits[i, :], log_pmf, where=elems)
        # normalise
        splits[i, :] -= hi
    return splits

def _split_limit(len1, len2, maxima1, len_bad_places, cost_ratio):
    Q = max(len1, len2) if len2 is not None else len1
    nc_cost = Q * len_bad_places
    split_ratio = nc_cost / (Q * np.log2(Q) * cost_ratio)
    if len2 is None:
        # we're squaring, so both sides have the same number of maxima
        lim = np.sqrt(split_ratio)
    else:
        # worst case the RHS will be 1
        other_split = 1 if maxima1 is None else maxima1
        lim = split_ratio / other_split
    logging.debug('computed split limit for %s %s %s %s %s as %s',
                  len1, len2, maxima1, len_bad_places, cost_ratio, lim)
    return lim

def _maxima_to_len(maxima):
    if isinstance(maxima, int) and maxima in (1, 2):
        return maxima, 'post-FFT-C estimated'
    else:
        return len(maxima), 'psfft computed'
def _is_nc_faster(len1, maxima1,
                  len2, maxima2,
                  len_bad_places,
                  cost_ratio):
    if len_bad_places == 0:
        nc_is_better = True
    elif maxima1 is None or maxima2 is None:
        nc_is_better = True
        logging.debug('psfft didn\'t compute all maxima (%s, %s). Using psfft? False',
                      maxima1, maxima2)
    else:
        Q = max(len1, len2)
        nc_cost = min(Q * len_bad_places, len1 * len2)

        len1, msg = _maxima_to_len(maxima1)
        len2, _ = _maxima_to_len(maxima2)
        afftc_cost = len1 * len2 * Q * np.log2(Q)
        scaled_cost = afftc_cost * cost_ratio
        nc_is_better = scaled_cost > nc_cost

        logging.debug('%s %d & %d splits, with %d locations to recompute, '
                      'giving scaled cost %.2e (vs. NC cost %.2e). '
                      'Using psfft? %s',
                      msg,
                      len1, len2,
                      len_bad_places,
                      scaled_cost,
                      nc_cost,
                      not nc_is_better)
    return nc_is_better

def _use_nc_if_better(log_pmf1, maxima1,
                      log_pmf2, maxima2,
                      direct, bad_places,
                      cost_ratio):
    nc_is_better = _is_nc_faster(len(log_pmf1), maxima1,
                                 len(log_pmf2), maxima2,
                                 len(bad_places),
                                 cost_ratio)
    if nc_is_better:
        if len(bad_places) > 0:
            naive.convolve_naive_into(direct, bad_places,
                                      log_pmf1, log_pmf2)
        return True
    else:
        return False


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
        r1 = utils.log_dynamic_range_shifted(log_pmf1, theta)
        r2 = utils.log_dynamic_range_shifted(log_pmf2, theta)
        return r1 * r2
    return optimize.fminbound(f, -OPT_BOUND, OPT_BOUND)
