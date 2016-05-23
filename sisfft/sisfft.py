import numpy as np, scipy as sc
from numpy.fft import fft, ifft
from scipy import optimize
import logging
import afftc, naive, utils
from utils import NEG_INF, EPS
from timer import timer

OPT_BOUND = 1e10
THETA_LIMIT = 1e4

def conv_power(log_pmf, L, desired_beta, desired_delta):
    """Compute $log(exp(log_pmf)**L)$ with overall accuracy parameters
       $desired_beta$ and $desired_delta$."""
    if L == 0:
        return np.array([0.0])
    elif L == 1:
        return log_pmf

    beta, delta = _accurate_error_bounds(L, 1.0 / desired_beta, desired_delta)
    if desired_delta == 0:
        delta = None

    answer = np.array([0.0])
    pmf_power = log_pmf
    while True:
        need_to_add = L % 2 == 1
        L /= 2
        need_to_square = L > 0
        squared = False

        if need_to_add:
            if len(answer) == 1:
                answer = pmf_power
            else:
                if need_to_square:
                    answer, pmf_power = afftc.convolve_and_square(pmf_power, answer, beta, delta)
                    squared = True
                else:
                    answer = afftc.convolve(pmf_power, answer, beta, delta)

        if not need_to_square:
            break
        if not squared:
            pmf_power = afftc.convolve_square(pmf_power, beta, delta)


    return answer

def pvalue(log_pmf, s0, L, desired_beta):
    """Compute $log((exp(log_pmf)**L)[s0:])$, such that the relative error
       to the exact answer is less than or equal to $desired_beta$."""
    total_len, _ = utils.iterated_convolution_lengths(len(log_pmf), L)
    if s0 >= total_len:
        return NEG_INF

    _, p_lower_preshift, p_upper_preshift = _bounds(log_pmf, log_pmf, 0, 0.0,
                                                    s0, L, desired_beta)
    sfft_good_preshift, sfft_pval_preshift = _check_sfft_pvalue(p_lower_preshift,
                                                                p_upper_preshift,
                                                                desired_beta)
    if sfft_good_preshift:
        logging.debug(' pre-shift sfft worked %.20f', sfft_pval_preshift)
        return sfft_pval_preshift

    with timer('computing theta'):
        theta = _compute_theta(log_pmf, s0, L)
    logging.debug('raw theta %s', theta)

    # TODO: too-large or negative theta causes numerical instability,
    # so this is a huge hack
    theta = utils.clamp(theta, 0, THETA_LIMIT)
    shifted_pmf, log_mgf = utils.shift(log_pmf, theta)

    beta = desired_beta / 2.0
    with timer('bounds'):
        log_delta, p_lower, p_upper = _bounds(log_pmf, shifted_pmf, theta, log_mgf,
                                              s0, L, desired_beta)

    sfft_good, sfft_pval = _check_sfft_pvalue(p_lower, p_upper, desired_beta)

    logging.debug('theta %s, log_mgf %s, beta %s, log delta %s', theta, log_mgf, beta, log_delta)
    if sfft_good:
        logging.debug(' sfft worked %.20f', sfft_pval)
        return sfft_pval
    delta = np.exp(log_delta)

    conv = conv_power(shifted_pmf, L, beta, delta)

    pval = utils.log_sum(utils.unshift(conv, theta, (log_mgf, L))[s0:])
    logging.debug(' sis pvalue %.20f', pval)
    return pval

def _check_sfft_pvalue(p_lower, p_upper, desired_beta):
    sfft_pval = np.log(2) + p_lower + p_upper - np.logaddexp(p_upper, p_lower)
    sfft_accuracy = utils.logsubexp(p_upper, p_lower) - np.logaddexp(p_upper, p_lower)
    logging.debug('sfft pval %s (range %s -- %s, accuracy %s, needed %s)',
                  sfft_pval, p_lower, p_upper, sfft_accuracy, np.log(desired_beta))
    return sfft_accuracy < np.log(desired_beta), sfft_pval

def _bounds(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta):
    # things aren't happy if this is (too) negative
    assert theta >= -1.0

    if L != 2:
        log_f0, fft_len = naive.power_fft(shifted_pmf, L - 1)
        f0 = np.exp(log_f0)
        error_estimate = utils.sfft_error_threshold_factor(fft_len, L - 1)
        v_lower = np.log(np.maximum(f0 - error_estimate, 0.0))
        v_upper = np.log(f0 + error_estimate)
    else:
        v_lower = v_upper = shifted_pmf

    tail_sums = np.logaddexp.accumulate(log_pmf[::-1])[::-1]
    Q = len(log_pmf)
    Q1 = Q - 1

    def pval_estimate(v):
        limit = len(v)
        low = max(s0 - Q1, 0)
        mid = min(s0, limit)

        if theta != 0:
            v += -np.arange(limit) * theta
        if log_mgf != 0:
            v += (L - 1) * log_mgf

        q1 = utils.log_sum(v[low:mid] + tail_sums[s0 - low:s0 - mid:-1])
        q2 = utils.log_sum(v[mid:limit])
        q = np.logaddexp(q1, q2)
        return q

    p_lower = pval_estimate(v_lower)
    p_upper = pval_estimate(v_upper)

    factor = L * Q1 - s0 + 1
    # TODO: this has numerical stability issues:
    if abs(theta) < 1e-16:
        # lim_{ theta -> 0 }
        frac = -np.log(factor)
    elif theta > 0.0:
        # (1 - exp(-theta)) / (1 - exp(-factor theta))
        frac = utils.log1subexp(-theta) - utils.log1subexp(-factor * theta)
    else:
        # theta is negative, so we negate top and bottom, so
        # subtraction works
        frac = utils.logsubexp(-theta, 0.0) - utils.logsubexp(-factor * theta, 0.0)

    logging.debug('lower %s, upper %s, frac %s, factor %s', p_lower, p_upper, frac, factor)
    log_delta = p_lower + (theta * s0 - L * log_mgf) + frac + np.log(desired_beta / 2)
    return log_delta, p_lower, p_upper


def _compute_theta(log_pmf, s0, L):
    s0L = float(s0) / L
    def f(theta):
        return utils.log_mgf(log_pmf, theta) - theta * s0L
    return optimize.fminbound(f, -OPT_BOUND, OPT_BOUND)

def _accurate_error_bounds(L, beta, gamma):
    beta2 = (1 + beta)**(1.0 / (L - 1)) - 1

    accum = 0
    ijs = []
    Ljs = []
    for i in range(int(np.log2(L)) + 1):
        this = 2**i
        if L & this:
            accum += this
            ijs.append(i)
            Ljs.append(accum)

    Lj = lambda j: Ljs[j - 1]
    ij = lambda j: ijs[j - 1]
    r = lambda i: (1 + beta2)**(2**i - 1) - 1
    rbar = lambda j: (1 + beta2)**(Lj(j) - 1) - 1
    d = lambda i: sum(2**(i - k) * (1 + r(k)) for k in range(0, i))
    dbar = lambda j: (sum(d(ij(k)) for k in range(1, j + 1)) +
                      sum(2 + rbar(k) + r(k + 1) for k in range(1, j)))
    delta = gamma / dbar(len(Ljs))
    return beta2, delta

def _log_sfft_error_estimate(pmf, log_mgf, theta, s0, L):
    true_len, fft_len = utils.iterated_convolution_lengths(len(pmf), L)
    smax = true_len - 1
    return np.log(utils.sfft_error_threshold_factor(fft_len, L) * (smax - s0)) + (-theta * s0 + L * log_mgf)
