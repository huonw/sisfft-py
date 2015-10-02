import numpy as np

NEG_INF = -float('inf')
EPS = np.finfo(float).eps

def log_min_pos(log_pmf):
    return log_pmf[log_pmf > NEG_INF].min()
def log_dynamic_range(log_pmf):
    hi = log_sum(2.0 * log_pmf) / 2
    lo = log_min_pos(log_pmf)
    return hi - lo

def shift(log_pmf, theta):
    shifted = log_pmf + theta * np.arange(len(log_pmf))
    log_mgf = log_sum(shifted)
    shifted -= log_mgf
    return shifted, log_mgf

def unshift(convolved, theta, *mgfs):
    c = convolved - theta * np.arange(len(convolved))
    for (mgf, multiplicity) in mgfs:
        c += multiplicity * mgf
    return c

def logsub1exp(log_y):
    assert log_y < 0.0
    return np.log(1 - np.exp(log_y))

def log_sum(log_u):
    """Compute `log(sum(exp(log_u)))`"""
    if len(log_u) == 0:
        return NEG_INF

    max = np.max(log_u)
    if max == NEG_INF:
        return max
    else:
        return np.log(np.sum(np.exp(log_u - max))) + max


def log_mgf(log_pmf, theta):
    return log_sum(log_pmf + theta * np.arange(len(log_pmf)))

def error_threshold_factor(conv_len):
    if conv_len > 2**5:
        c = 13.5
    else:
        c = 16
    return EPS * c * np.log2(conv_len)


def _next_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))

def pairwise_convolution_lengths(a, b):
    true = a + b - 1
    return true, _next_power_of_two(true)

def iterated_convolution_lengths(a, L):
    true = (a - 1) * L + 1
    return true, _next_power_of_two(true)
