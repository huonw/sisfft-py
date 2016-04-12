import numpy as np

NEG_INF = -float('inf')
EPS = np.finfo(float).eps / 2

def clamp(x, lo, hi):
    return min(max(x, lo), hi)

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

def log_dynamic_range_shifted(log_pmf, theta):
    # this is equivalent to log_dynamic_range(shift(log_pmf,
    # theta)[0]), but is more efficient, as it avoids an unnecessary
    # log_sum computation.
    shifted = log_pmf + np.arange(float(len(log_pmf))) * theta
    lo = log_min_pos(shifted)
    hi = log_sum(shifted * 2.0) / 2
    return hi - lo

def unshift(convolved, theta, *mgfs):
    c = convolved - theta * np.arange(len(convolved))
    for (mgf, multiplicity) in mgfs:
        c += multiplicity * mgf
    return c

def logsubexp(log_x, log_y):
    assert log_x >= log_y
    return log_x + np.log1p(-np.exp(log_y - log_x))

def log1subexp(log_y):
    assert log_y <= 0.0
    return np.log1p(-np.exp(log_y))

def log_sum(log_u):
    """Compute `log(sum(exp(log_u)))`"""
    if len(log_u) == 0:
        return NEG_INF

    maxi = np.argmax(log_u)
    max = log_u[maxi]
    if max == NEG_INF:
        return max
    else:
        exp = log_u - max
        np.exp(exp, out = exp)
        return np.log1p(np.sum(exp[:maxi]) + np.sum(exp[maxi + 1:])) + max


def log_mgf(log_pmf, theta):
    return log_sum(log_pmf + theta * np.arange(len(log_pmf)))

def error_threshold_factor(conv_len):
    if conv_len > 2**5:
        c = 13.5
    else:
        c = 16
    return EPS * c * np.log2(conv_len)
def sfft_error_threshold_factor(conv_len, L):
    if conv_len >= 2**6 and L >= 10:
        c = 5
    else:
        c = 7
    return EPS * c * np.log2(conv_len) * L

def _next_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))

def pairwise_convolution_lengths(a, b):
    true = a + b - 1
    return true, _next_power_of_two(true)

def iterated_convolution_lengths(a, L):
    true = (a - 1) * L + 1
    return true, _next_power_of_two(true)
