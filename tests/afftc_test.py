import numpy as np
import unittest
from sisfft.timer import timer
import sisfft
afftc = sisfft._afftc

#logging.basicConfig(level = logging.INFO)

TEST_REPEATS = 20
TEST_LENGTH = 100

def afftc_(self, beta, delta):
    np.random.seed(1)
    for _ in range(0, TEST_REPEATS):
        v1 = np.random.rand(TEST_LENGTH)
        v1 /= v1.sum()
        v2 = np.random.rand(TEST_LENGTH)
        v2 /= v2.sum()

        with timer('naive'):
            real = np.convolve(v1, v2)
        with timer('afftc'):
            hopeful = np.exp(afftc.convolve(np.log(v1), np.log(v2), beta, delta))

        lower = (1 - beta) * real - 2 * delta
        upper = (1 + beta) * real
        between = (lower <= hopeful) & (hopeful <= upper)
        not_between = np.invert(between)
        self.assertTrue(between.all(),
                        '%s\n%s' % (hopeful[not_between], real[not_between]))

def afftc_square_(self, beta, delta):
    np.random.seed(1)
    for _ in range(0, TEST_REPEATS):
        v = np.random.rand(TEST_LENGTH)
        v /= v.sum()

        with timer('naive'):
            real = np.convolve(v, v)
        with timer('afftc'):
            hopeful = np.exp(afftc.convolve_square(np.log(v), beta, delta))

        lower = (1 - beta) * real - 2 * delta
        upper = (1 + beta) * real
        between = (lower <= hopeful) & (hopeful <= upper)
        not_between = np.invert(between)
        self.assertTrue(between.all(),
                        '%s\n%s' % (hopeful[not_between], real[not_between]))

def afftc_no_lower_bound_(self, beta):
    np.random.seed(1)
    for _ in range(0, TEST_REPEATS):
        v1 = np.random.rand(TEST_LENGTH)
        v1 /= v1.sum()
        v2 = np.random.rand(TEST_LENGTH)
        v2 /= v2.sum()

        with timer('naive'):
            real = np.convolve(v1, v2)
        with timer('afftc'):
            hopeful = np.exp(afftc.convolve(np.log(v1), np.log(v2), beta, None))

        lower = (1 - beta) * real
        upper = (1 + beta) * real
        between = (lower <= hopeful) & (hopeful <= upper)
        not_between = np.invert(between)
        self.assertTrue(between.all(),
                        '%s\n%s' % (hopeful[not_between], real[not_between]))

class Afftc(unittest.TestCase):
    pass

for log10_alpha in range(1, 5 + 1):
    beta = 10**-log10_alpha
    for delta in [0.0, 0.0001, 0.01, 0.1]:
        test = lambda self, beta=beta, delta=delta: afftc_(self, beta, delta)
        test_square = lambda self, beta=beta, delta=delta: afftc_square_(self, beta, delta)

        name = 'test_%s_%f' % (beta, delta)
        name_square = 'test_square_%s_%f' % (beta, delta)

        test.__name__ = name
        test_square.__name__ = name_square
        setattr(Afftc, name, test)
        setattr(Afftc, name_square, test_square)
        del test, test_square

    name_no_bound = 'test_no_bound_%s' % beta
    test_no_bound = lambda self, beta=beta: afftc_no_lower_bound_(self, beta)
    test_no_bound.__name__ = name_no_bound
    setattr(Afftc, name_no_bound, test_no_bound)
    del test_no_bound
