try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'sisfft',
    'version': '0.1',
    'test_suite': 'tests',
}

setup(**config)
