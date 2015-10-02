import numpy as np
import unittest
import logging
from sisfft.timer import timer
from sisfft import sisfft, utils, naive

TEST_REPEATS = 1 # too slow
TEST_LENGTH = 100

def power_naive(v, L):
    answer = np.array([1])
    power = v
    while L > 0:
        if L % 2 == 1:
            if len(answer) == 1:
                answer = power
            else:
                answer = np.convolve(answer, power)

        L /= 2
        power = np.convolve(power, power)
    return answer
def log_power_naive(v, L):
    answer = np.array([1])
    power = v
    while L > 0:
        if L % 2 == 1:
            if len(answer) == 1:
                answer = power
            else:
                answer = naive.convolve_naive(answer, power)

        L /= 2
        power = naive.convolve_naive(power, power)
    return answer

class ConvPower(unittest.TestCase):
    def conv_power_test(self, alpha, delta, L):
        for _ in range(0, TEST_REPEATS):
            v = np.random.rand(TEST_LENGTH)
            v /= v.sum()

            with timer('naive'):
                real = power_naive(v, L)
            with timer('sisfft'):
                hopeful = np.exp(sisfft.conv_power(np.log(v), L, alpha, delta))

            lower = (1 - 1.0 / alpha) * real - delta
            upper = (1 + 1.0 / alpha) * real
            between = (lower <= hopeful) & (hopeful <= upper)
            not_between = np.invert(between)
            self.assertTrue(between.all(),
                            '%s\n%s' % (hopeful[not_between], real[not_between]))
class Sisfft(unittest.TestCase):
    def sisfft_test(self, beta, s0, L):
        for _ in range(0, TEST_REPEATS):
            v = np.random.rand(TEST_LENGTH)
            v /= v.sum()

            with timer('naive'):
                real = utils.log_sum(log_power_naive(np.log(v), L)[s0:])
            with timer('sisfft'):
                hopeful = sisfft.pvalue(np.log(v), s0, L, beta)
            logging.debug('true pvalue %.20f', real)
            abs_diff = utils.logsubexp(max(real, hopeful), min(real, hopeful))
            self.assertLessEqual(abs_diff, np.log(beta) + real,
                                 '%s isn\'t close to %s' % (hopeful, real))



for log10_alpha in range(1, 9 + 1):
    alpha = 10.0**log10_alpha
    for logL in range(1, 8 + 1, 1):
        for L in [2**logL, 2**logL - 1, 2**logL + 1]:
            # for L in range(2, 59, 7):
            for delta in [0.0001, 0.01, 0.1]:
                test = lambda self, alpha=alpha, delta=delta, L=L: self.conv_power_test(alpha, delta, L)

                name = 'test_conv_power_%s_%f_%s' % (alpha, delta, L)

                setattr(ConvPower, name, test)

            for s0_ratio in [0.5, 0.9, 0.99]:
                s0 = int(s0_ratio * utils.iterated_convolution_lengths(TEST_LENGTH, L)[0])
                beta = 1/alpha

                test = lambda self, beta = beta, s0=s0, L=L: self.sisfft_test(beta, s0, L)
                name = 'test_%.9f_%04d_%04d' % (beta, L, s0)
                setattr(Sisfft, name, test)

