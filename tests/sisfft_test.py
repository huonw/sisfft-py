import numpy as np
import unittest
from sisfft.timer import timer
from sisfft import sisfft

np.random.seed(1)

TEST_REPEATS = 10
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

class Sisfft(unittest.TestCase):
    def sisfft_test(self, alpha, delta, L):
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


for log10_alpha in range(1, 5 + 1):
    alpha = 10**log10_alpha
    for delta in [0.0001, 0.01, 0.1]:
        for L in range(2, 10):
            test = lambda self, alpha=alpha, delta=delta, L=L: self.sisfft_test(alpha, delta, L)

            name = 'test_%s_%s_%s' % (alpha, delta, L)

            setattr(Sisfft, name, test)

