import numpy as np, scipy as sc
from numpy.fft import fft, ifft
from scipy import optimize
import logging
import psfft, naive, utils
from utils import NEG_INF

OPT_BOUND = 1e10
THETA_LIMIT = 1e4

def conv_power(log_pmf, L, desired_alpha, desired_delta, accurate_bounds = True):
    if L == 0:
        return np.array([0.0])
    elif L == 1:
        return log_pmf

    if accurate_bounds:
        alpha, delta = _accurate_error_bounds(L, 1.0 / desired_alpha, desired_delta)
    else:
        alpha, delta = _estimate_error_bounds(L, 1.0 / desired_alpha, desired_delta)

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
                    answer, pmf_power = psfft.convolve_and_square(pmf_power, answer, alpha, delta)
                    squared = True
                else:
                    answer = psfft.convolve(pmf_power, answer, alpha, delta)

        if not need_to_square:
            break
        if not squared:
            pmf_power = psfft.convolve_square(pmf_power, alpha, delta)


    return answer

def pvalue(log_pmf, s0, L, desired_beta, accurate_bounds = True):
    total_len, _ = utils.iterated_convolution_lengths(len(log_pmf), L)
    if s0 >= total_len:
        return NEG_INF

    theta = _compute_theta(log_pmf, s0, L)
    logging.debug('raw theta %s', theta)
    if theta < 0.0:
        logging.debug('    computing right tail')
        # turn things around! Compute 1 - sum(left tail) instead of sum(right tail).
        p = pvalue(log_pmf[::-1], total_len - s0, L, desired_beta, accurate_bounds)
        return utils.log1subexp(p)

    # TODO: too-large theta causes numerical instability, so this is a
    # huge hack
    theta = utils.clamp(theta, -THETA_LIMIT, THETA_LIMIT)
    shifted_pmf, log_mgf = utils.shift(log_pmf, theta)

    alpha = 2.0 / desired_beta
    log_delta = _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta)
    logging.debug('theta %s, log_mgf %s, alpha %s, log delta %s', theta, log_mgf, alpha, log_delta)
    delta = np.exp(log_delta)

    conv = conv_power(shifted_pmf, L, alpha, delta, accurate_bounds)

    pval = utils.log_sum(utils.unshift(conv, theta, (log_mgf, L))[s0:])
    logging.debug(' sis pvalue %.20f', pval)
    return pval

def _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta):
    # things aren't happy if this is negative
    assert theta >= 0.0

    log_f0, fft_len = naive.power_fft(shifted_pmf, L - 1)
    f0 = np.exp(log_f0)
    error_estimate = utils.error_threshold_factor(fft_len) * (L - 1)

    f_theta = np.where(f0 > error_estimate,
                       f0 - error_estimate,
                       0.0)
    f_theta = np.log(f_theta)

    tail_sums = np.zeros_like(log_pmf)
    tail_sums[-1] = log_pmf[-1]
    Q = len(log_pmf)
    Q1 = Q - 1
    for i in range(1, Q):
        tail_sums[-1 - i] = np.logaddexp(log_pmf[-1 - i], tail_sums[-i])

    limit = len(f_theta)
    low = max(s0 - Q1, 0)
    k1 = np.arange(low, min(s0, limit))
    q1 = utils.log_sum(f_theta[k1] + (-k1 * theta + (L - 1) * log_mgf) + tail_sums[s0 - k1])

    k2 = np.arange(min(s0, limit), limit)
    q2 = utils.log_sum(f_theta[k2] + (-k2 * theta + (L - 1) * log_mgf))
    q = np.logaddexp(q1, q2)

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
        # this shouldn't be reached at the moment
        raise NotImplementedError()

    logging.debug('q %s, frac %s, factor %s', q, frac, factor)
    gamma = q + (theta * s0 - L * log_mgf) + frac + np.log(desired_beta / 2)
    return gamma


def _compute_theta(log_pmf, s0, L):
    s0L = float(s0) / L
    def f(theta):
        return utils.log_mgf(log_pmf, theta) - theta * s0L
    return optimize.fminbound(f, -OPT_BOUND, OPT_BOUND)

def _estimate_error_bounds(L, beta, gamma):
    alpha = 2.0 * (L - 1) / beta
    delta = gamma / (2.0 * (L - 1))
    return (alpha, delta)

def _accurate_error_bounds(L, beta, gamma):
    est_alpha, est_delta = _estimate_error_bounds(L, beta, gamma)

    def compute_pairwise(inv_alpha, delta):
        vals = [(0.0, 0.0)]
        k = 1
        while L >= k:
            lastr, lastd = vals[-1]
            r = lastr * (2 + lastr) * (1 + inv_alpha) + inv_alpha
            d = 2 * (delta * (1 + lastr) + lastd)
            vals.append((r, d))
            k *= 2
        return vals

    def compute_iterated(alpha, delta):
        inv_alpha = 1.0 / alpha
        vals = compute_pairwise(inv_alpha, delta)
        rbar = dbar = None
        check = 0
        for j, (r, d) in enumerate(vals):
            if L & (1 << j) != 0:
                check += 1 << j
                if rbar is None:
                    rbar = r
                    dbar = d
                else:
                    # have to use the old rbar
                    dbar = 2 * delta + dbar + d + delta * (rbar + r)
                    rbar = inv_alpha + (1 + inv_alpha) * (rbar + r + rbar * r)
        assert check == L
        return rbar, dbar

    def compute_rbar(alpha):
        rbar, _ = compute_iterated(alpha, 0.0)
        return rbar - beta / 2.0

    alpha = optimize.fsolve(compute_rbar, est_alpha)[0]

    def compute_deltabar(delta):
        _, dbar = compute_iterated(alpha, delta)
        return dbar - gamma

    delta = optimize.fsolve(compute_deltabar, est_delta)[0]

    logging.debug('computed accurate error bounds: alpha %.20f (vs. %.20f), delta %.10g (vs. %.10g)',
                  alpha, est_alpha,
                  delta, est_delta)
    assert alpha >= est_alpha
    assert delta <= est_delta
    return (alpha, delta)
