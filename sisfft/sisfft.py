import numpy as np, scipy as sc
from numpy.fft import fft, ifft
from scipy import optimize
import logging
import afftc, naive, utils
from utils import NEG_INF

OPT_BOUND = 1e10
THETA_LIMIT = 1e4

def conv_power(log_pmf, L, desired_alpha, desired_delta):
    if L == 0:
        return np.array([0.0])
    elif L == 1:
        return log_pmf

    alpha, delta = _accurate_error_bounds(L, 1.0 / desired_alpha, desired_delta)

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
                    answer, pmf_power = afftc.convolve_and_square(pmf_power, answer, alpha, delta)
                    squared = True
                else:
                    answer = afftc.convolve(pmf_power, answer, alpha, delta)

        if not need_to_square:
            break
        if not squared:
            pmf_power = afftc.convolve_square(pmf_power, alpha, delta)


    return answer

def pvalue(log_pmf, s0, L, desired_beta):
    total_len, _ = utils.iterated_convolution_lengths(len(log_pmf), L)
    if s0 >= total_len:
        return NEG_INF

    theta = _compute_theta(log_pmf, s0, L)
    logging.debug('raw theta %s', theta)
    # theoretically this could/should be < 0.0, but that leads to
    # infinite recursion (there are vectors for which theta is
    # computed to be negative for both configurations)
    if theta < -1:
        logging.debug('    computing right tail')
        # turn things around! Compute 1 - sum(left tail) instead of sum(right tail).
        p = pvalue(log_pmf[::-1], total_len - s0, L, desired_beta)
        return utils.log1subexp(p)

    # TODO: too-large theta causes numerical instability, so this is a
    # huge hack
    theta = utils.clamp(theta, -THETA_LIMIT, THETA_LIMIT)
    shifted_pmf, log_mgf = utils.shift(log_pmf, theta)

    alpha = 2.0 / desired_beta
    log_delta = _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta)
    logging.debug('theta %s, log_mgf %s, alpha %s, log delta %s', theta, log_mgf, alpha, log_delta)
    delta = np.exp(log_delta)

    conv = conv_power(shifted_pmf, L, alpha, delta)

    pval = utils.log_sum(utils.unshift(conv, theta, (log_mgf, L))[s0:])
    logging.debug(' sis pvalue %.20f', pval)
    return pval

def _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta):
    # things aren't happy if this is (too) negative
    assert theta >= -1.0

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

    logging.debug('q %s, frac %s, factor %s', q, frac, factor)
    gamma = q + (theta * s0 - L * log_mgf) + frac + np.log(desired_beta / 2)
    return gamma


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
    return 1.0 / beta2, delta
