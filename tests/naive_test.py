import numpy as np
import unittest
from sisfft.timer import timer
from sisfft import naive

TEST_REPEATS = 10
TEST_LENGTH = 100

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
