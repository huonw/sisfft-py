import numpy as np
import sisfft, utils, naive
from timer import timer

def sfft_pvalue(log_pmf, s0, L):
    theta = sisfft._compute_theta(log_pmf, s0, L)
    shifted, mgf = utils.shift(log_pmf, theta)
    sfft_vector, fft_len = naive.power_fft(shifted, L)
    error_estimate = utils.sfft_error_threshold_factor(fft_len, L)
    sfft_vector[sfft_vector < np.log(error_estimate)] = utils.NEG_INF
    return utils.log_sum(utils.unshift(sfft_vector, theta, (mgf, L))[s0:])
