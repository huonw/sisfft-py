import numpy as np
import unittest
from sisfft.timer import timer
from sisfft import psfft

#logging.basicConfig(level = logging.INFO)

TEST_REPEATS = 20
TEST_LENGTH = 100

def psfft_(self, alpha, delta):
    np.random.seed(1)
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

def psfft_square_(self, alpha, delta):
    np.random.seed(1)
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

def psfft_no_lower_bound_(self, alpha):
    np.random.seed(1)
    for _ in range(0, TEST_REPEATS):
        v1 = np.random.rand(TEST_LENGTH)
        v1 /= v1.sum()
        v2 = np.random.rand(TEST_LENGTH)
        v2 /= v2.sum()

        with timer('naive'):
            real = np.convolve(v1, v2)
        with timer('psfft'):
            hopeful = np.exp(psfft.convolve(np.log(v1), np.log(v2), alpha, None))

        lower = (1 - 1.0 / alpha) * real
        upper = (1 + 1.0 / alpha) * real
        between = (lower <= hopeful) & (hopeful <= upper)
        not_between = np.invert(between)
        self.assertTrue(between.all(),
                        '%s\n%s' % (hopeful[not_between], real[not_between]))

class Psfft(unittest.TestCase):
    pass

for log10_alpha in range(1, 5 + 1):
    alpha = 10**log10_alpha
    for delta in [0.0, 0.0001, 0.01, 0.1]:
        test = lambda self, alpha=alpha, delta=delta: psfft_(self, alpha, delta)
        test_square = lambda self, alpha=alpha, delta=delta: psfft_square_(self, alpha, delta)

        name = 'test_%s_%f' % (alpha, delta)
        name_square = 'test_square_%s_%f' % (alpha, delta)

        test.__name__ = name
        test_square.__name__ = name_square
        setattr(Psfft, name, test)
        setattr(Psfft, name_square, test_square)
        del test, test_square

    name_no_bound = 'test_no_bound_%s' % alpha
    test_no_bound = lambda self, alpha=alpha: psfft_no_lower_bound_(self, alpha)
    test_no_bound.__name__ = name_no_bound
    setattr(Psfft, name_no_bound, test_no_bound)
    del test_no_bound
