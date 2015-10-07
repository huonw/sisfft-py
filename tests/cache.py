import sqlite3
import functools
import inspect
import hashlib
import os, errno
import logging
from os import path

import numpy

def make_dirs(p):
    try:
        os.makedirs(p)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return
        else:
            raise


PREFIX = 'sisfft-cached-tests'
make_dirs(PREFIX)

def disk_memo(f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        dirname = path.join(PREFIX, f.__name__)
        fname = 'f'
        for a in args:
            if isinstance(a, numpy.ndarray):
                # hash a numpy array, don't use it directly, because
                # it could be quite long
                h = hashlib.sha256()
                h.update(a.data)
                fname += '_' + h.hexdigest()
            else:
                # use everything else in the directory name, so
                # finding the relevant files for a GC is easier
                dirname = path.join(dirname, repr(a))
        for k,v in kwargs.items():
            if isinstance(v, numpy.ndarray):
                h = hashlib.sha256()
                h.update(a.data)
                fname += '_%s=%s' % (k, h.hexdigest())
            else:
                dirname = path.join(dirname, '%s=%s' % (k, repr(v)))

        fname += '.npy'
        logging.info('looking in %s for %s' % (dirname, fname))
        make_dirs(dirname)
        full_path = path.join(dirname, fname)
        try:
            loaded = numpy.load(full_path)
            logging.info('    found')
            return loaded
        except:
            logging.info('    not found, recomputing')
            value = f(*args, **kwargs)
            numpy.save(full_path, value)
            return value
    if os.environ.get('SISFFT_NO_CACHE', '0') == '1':
        return f
    else:
        return g
