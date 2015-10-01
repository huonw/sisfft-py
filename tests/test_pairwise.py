import numpy as np
import unittest
from sisfft.timer import timer
from sisfft import psfft, naive

np.random.seed(1)
#logging.basicConfig(level = logging.INFO)

TEST_REPEATS = 20
TEST_LENGTH = 100

class Psfft(unittest.TestCase):
    def psfft_test(self, alpha, delta):
        for _ in range(0, TEST_REPEATS):
            v1 = np.random.rand(TEST_LENGTH)
            v1 /= v1.sum()
            v2 = np.random.rand(TEST_LENGTH)
            v2 /= v2.sum()

            with timer('naive'):
                real = np.convolve(v1, v2)
            with timer('psfft'):
                hopeful = np.exp(psfft.convolve(np.log(v1), np.log(v2), alpha, delta))

            lower = (1 - 1.0 / alpha) * real - 2 * delta
            upper = (1 + 1.0 / alpha) * real
            between = (lower <= hopeful) & (hopeful <= upper)
            not_between = np.invert(between)
            self.assertTrue(between.all(),
                            '%s\n%s' % (hopeful[not_between], real[not_between]))

    def psfft_square_test(self, alpha, delta):
        for _ in range(0, TEST_REPEATS):
            v = np.random.rand(TEST_LENGTH)
            v /= v.sum()

            with timer('naive'):
                real = np.convolve(v, v)
            with timer('psfft'):
                hopeful = np.exp(psfft.convolve_square(np.log(v), alpha, delta))

            lower = (1 - 1.0 / alpha) * real - 2 * delta
            upper = (1 + 1.0 / alpha) * real
            between = (lower <= hopeful) & (hopeful <= upper)
            not_between = np.invert(between)
            self.assertTrue(between.all(),
                            '%s\n%s' % (hopeful[not_between], real[not_between]))

for log10_alpha in range(1, 5 + 1):
    alpha = 10**log10_alpha
    for delta in [0.0, 0.0001, 0.01, 0.1]:
        test = lambda self, alpha=alpha, delta=delta: self.psfft_test(alpha, delta)
        test_square = lambda self, alpha=alpha, delta=delta: self.psfft_square_test(alpha, delta)

        name = 'test_%s_%s' % (alpha, delta)
        name_square = 'test_square_%s_%s' % (alpha, delta)

        setattr(Psfft, name, test)
        setattr(Psfft, name_square, test_square)

class Naive(unittest.TestCase):
    def test_naive(self):
        for _ in range(0, TEST_REPEATS):
            v1 = np.random.rand(TEST_LENGTH)
            v2 = np.random.rand(TEST_LENGTH)

            with timer('numpy'):
                real = np.convolve(v1, v2)
            with timer('log'):
                hopeful = np.exp(naive.convolve_naive(np.log(v1), np.log(v2)))
            self.assertTrue(np.allclose(real, hopeful))
