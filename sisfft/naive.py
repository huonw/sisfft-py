import numpy as np
from timer import timer
import unittest

from utils import NEG_INF, log_sum

def convolve_naive(log_u, log_v):
    nu = len(log_u)
    nv = len(log_v)
    nc = nu + nv - 1

    log_c = np.repeat(NEG_INF, nc)

    for k in range(0, nc):
        low_j = max(0, k - nv + 1)
        hi_j = min(k + 1, nu)
        slice_u = log_u[low_j:hi_j]
        slice_v = log_v[k - low_j:k - hi_j:-1] if k - hi_j != -1 else log_v[k - low_j::-1]
        log_c[k] = log_sum(slice_u + slice_v)
    return log_c


