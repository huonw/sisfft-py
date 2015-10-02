import numpy as np, scipy as sc
from numpy.fft import fft, ifft
from scipy import optimize
import logging
import psfft, naive, utils
from utils import NEG_INF

OPT_BOUND = 1e10
THETA_LIMIT = 1e4

def conv_power(log_pmf, L, desired_alpha, desired_delta):
    if L == 0:
        return np.array([0.0])
    elif L == 1:
        return log_pmf

    alpha = (L - 1) * desired_alpha
    delta = desired_delta / (2.0 * (L - 1))

    answer = np.array([0.0])
    pmf_power = log_pmf
    while L > 0:
        if L % 2 == 1:
            if len(answer) == 1:
                answer = pmf_power
            else:
                x = pmf_power
                y = answer
                answer = psfft.convolve(x, y, alpha, delta)[:len(answer) + len(pmf_power) - 1]

        L /= 2
        pmf_power = psfft.convolve_square(pmf_power, alpha, delta)


    return answer

def pvalue(log_pmf, s0, L, desired_beta):
    theta = _compute_theta(log_pmf, s0, L)
    # TODO: too-large theta causes numerical instability, so this is a
    # huge hack
    theta = utils.clamp(theta, -THETA_LIMIT, THETA_LIMIT)
    shifted_pmf, log_mgf = utils.shift(log_pmf, theta)

    alpha = 2.0 / desired_beta
    delta = _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta)
    logging.debug('theta %s, log_mgf %s, alpha %s, log delta %s', theta, log_mgf, alpha, np.log(delta))
    if s0 == 230:
        import pdb; pdb.set_trace()

    conv = conv_power(shifted_pmf, L, alpha, delta)

    pval = utils.log_sum(utils.unshift(conv, theta, (log_mgf, L))[s0:])
    logging.debug(' sis pvalue %.20f', pval)
    return pval

def _lower_bound(log_pmf, shifted_pmf, theta, log_mgf, s0, L, desired_beta):
    f0 = naive.power_fft(shifted_pmf, L - 1)
    error_estimate = np.log(utils.error_threshold_factor(len(f0)) * (L - 1))

    f_theta = np.where(f0 > error_estimate / desired_beta,
                       f0 - error_estimate,
                       NEG_INF)

    tail_sums = np.zeros_like(log_pmf)
    tail_sums[-1] = log_pmf[-1]
    Q = len(log_pmf)
    Q1 = Q - 1
    for i in range(1, Q):
        tail_sums[-1 - i] = np.logaddexp(log_pmf[-1 - i], tail_sums[-i])

    limit = len(f0)
    low = max(s0 - Q1, 0)
    k1 = np.arange(low, min(s0, limit))
    q1 = utils.log_sum(f0[k1] + (-k1 * theta + (L - 1) * log_mgf) + tail_sums[s0 - k1])

    k2 = np.arange(min(s0, limit), limit)
    q2 = utils.log_sum(f0[k2] + (-k2 * theta + (L - 1) * log_mgf))
    q = np.logaddexp(q1, q2)

    factor = L * Q1 - s0 + 1
    # TODO: this has numerical stability issues:
    if abs(theta) < 1e-16:
        # lim_{ theta -> 0 }
        frac = -np.log(factor)
    elif theta > 0.0:
        frac = utils.logsub1exp(-theta) - utils.logsub1exp(-factor * theta)
    else:
        frac = utils.logsubexp(-theta, 0.0) + (factor - 1) * theta - utils.logsubexp(-factor * theta, 0.0)

    logging.debug('frac %s', frac)
    gamma = np.exp(q + (theta * s0 - L * log_mgf) + frac) * desired_beta / 2
    return gamma


def _compute_theta(log_pmf, s0, L):
    s0L = float(s0) / L
    def f(theta):
        return utils.log_mgf(log_pmf, theta) - theta * s0L
    return optimize.fminbound(f, -OPT_BOUND, OPT_BOUND)
