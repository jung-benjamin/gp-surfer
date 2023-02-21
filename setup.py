from setuptools import setup, find_packages

setup(
    name = 'gaussianprocesses',
    version = '0.4.0',
    packages = find_packages(include = ['gaussianprocesses', 'gaussianprocesses.*']),
    description = 'A package for gaussian process regression.',
    entry_points={'console_scripts': ['train_gp_surrogates = scripts:train_gp_surrogates']},
    )
