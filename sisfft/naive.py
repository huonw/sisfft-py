import numpy as np
from numpy import fft
from timer import timer
import unittest

from utils import NEG_INF
import utils

def convolve_naive(log_u, log_v):
    nu = len(log_u)
    nv = len(log_v)
    nc = nu + nv - 1

    log_c = np.repeat(NEG_INF, nc)
    convolve_naive_into(log_c, range(0, nc), log_u, log_v)
    return log_c

def convolve_naive_into(log_c, locations, log_u, log_v):
    nu = len(log_u)
    nv = len(log_v)
    nc = nu + nv - 1
    assert len(log_c) == nc

    for k in locations:
        low_j = max(0, k - nv + 1)
        hi_j = min(k + 1, nu)
        slice_u = log_u[low_j:hi_j]
        slice_v = log_v[k - low_j:k - hi_j:-1] if k - hi_j != -1 else log_v[k - low_j::-1]
        log_c[k] = utils.log_sum(slice_u + slice_v)

def convolve_fft(log_u, log_v):
    true_len, fft_len = utils.pairwise_convolution_lengths(len(log_u), len(log_v))
    fft1 = fft.fft(np.exp(log_u), n = fft_len)
    fft2 = fft.fft(np.exp(log_v), n = fft_len)
    conv = fft.ifft(fft1 * fft2)[:true_len]
    return np.log(np.abs(conv))

def power_naive(log_v, L):
    answer = np.array([0.0])
    if L == 0:
        return answer

    power = log_v
    while True:
        if L % 2 == 1:
            if len(answer) == 1:
                answer = power
            else:
                answer = convolve_naive(answer, power)

        L /= 2
        if L == 0:
             break
        power = convolve_naive(power, power)
    return answer

def power_fft(log_u, L):
    if L == 1:
        return log_u, 1

    true_len, fft_len = utils.iterated_convolution_lengths(len(log_u), L)
    fft_ = fft.fft(np.exp(log_u), n = fft_len)
    conv = fft.ifft(fft_**L)[:true_len]
    return np.log(np.abs(conv)), fft_len
