from __future__ import print_function
import contextlib, logging, sys
from timeit import default_timer

LIMIT = 3

@contextlib.contextmanager
def timer_real(name):
    start = default_timer()
    timer_real.depth += 1
    yield
    timer_real.depth -= 1
    end = default_timer()
    if timer_real.depth < LIMIT:
        logging.info('%s%s: %.3fms', '  ' * timer_real.depth, name, (end - start) * 1000)
timer_real.depth = 0

@contextlib.contextmanager
def timer_fake(name):
    # don't need to time/print anything
    yield

if 1:
    timer = timer_real
else:
    timer = timer_fake
