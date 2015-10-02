import numpy as np, scipy as sc
from numpy.fft import fft, ifft
from scipy import optimize

import psfft, naive, utils

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


