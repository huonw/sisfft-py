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

def log_sum(log_u):
    """Compute `log(sum(exp(log_u)))`"""
    if len(log_u) == 0:
        return NEG_INF

    max = np.max(log_u)
    if max == NEG_INF:
        return max
    else:
        return np.log(np.sum(np.exp(log_u - max))) + max

