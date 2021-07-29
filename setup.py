from setuptools import setup, find_packages

setup(
    name = 'gaussianprocesses',
    version = '0.1.0dev',
    packages = find_packages(include = ['gaussianprocesses', 'gaussianprocesses.*']),
    description = 'A package for gaussian process regression with the theano backend.',
    )
